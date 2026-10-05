"""Prepare and review manually identified image correspondences; no SfM.

Run with the repository's .venv/bin/python. Use --manual to reselect points.
The JSON selection file contains full-resolution upright (x, y) coordinates.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import textwrap

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

BASE = Path(__file__).resolve().parent
NAMES = ("IMG_7596.jpeg", "IMG_7597.jpeg", "IMG_7598.jpeg", "IMG_7603.jpeg")
CONVENTION = "Full-resolution upright pixels; origin upper-left; +x right; +y down"


def hashes():
    return {name: hashlib.sha256((BASE / "input" / name).read_bytes()).hexdigest()
            for name in NAMES}


def font(size, bold=False):
    names = ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", "DejaVuSans-Bold.ttf") if bold else (
        "/System/Library/Fonts/Supplemental/Arial.ttf", "DejaVuSans.ttf")
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def prepare():
    (BASE / "working").mkdir(exist_ok=True)
    images, metadata = [], []
    for view, name in enumerate(NAMES, 1):
        with Image.open(BASE / "input" / name) as source:
            orientation = source.getexif().get(274, 1)
            image = ImageOps.exif_transpose(source).convert("RGB")
            # Upright pixel data only: avoid a second EXIF rotation by readers.
            image.info.clear()
            destination = BASE / "working" / f"view_{view}_upright.png"
            image.save(destination)
        with Image.open(destination) as saved:
            saved.load()
            assert saved.size == image.size
        loaded = cv2.imread(str(destination), cv2.IMREAD_UNCHANGED)
        if loaded is None or loaded.shape != (image.height, image.width, 3):
            raise ValueError(f"Working image failed load verification: {destination}")
        images.append(image)
        metadata.append(dict(view=view, original=name, exif_orientation=orientation,
                             width=image.width, height=image.height,
                             opencv_shape=list(loaded.shape)))
    return images, metadata


def validate(points, images):
    if not 8 <= len(points) <= 12:
        raise ValueError("Select 8–12 points")
    expected = [f"P{i}" for i in range(1, len(points) + 1)]
    if [p["point_id"] for p in points] != expected:
        raise ValueError("Point IDs must be unique and ordered P1, P2, ...")
    corners = ["Front cover top-left corner", "Front cover top-right corner",
               "Front cover bottom-right corner", "Front cover bottom-left corner"]
    if [p["description"] for p in points[:4]] != corners:
        raise ValueError("P1–P4 must retain the physical front-cover corner identities")
    seen = [set() for _ in images]
    for point in points:
        if not point["description"] or len(point["views"]) != 4:
            raise ValueError("Each point needs a description and all four views")
        for v, (xy, image) in enumerate(zip(point["views"], images)):
            if len(xy) != 2 or not all(isinstance(n, (int, float)) and math.isfinite(n) for n in xy):
                raise ValueError(f"Missing/nonfinite coordinate: {point['point_id']}")
            x, y = xy
            if not (0 <= x < image.width and 0 <= y < image.height):
                raise ValueError(f"Coordinate outside View {v + 1}: {point['point_id']}")
            if tuple(xy) in seen[v]:
                raise ValueError(f"Duplicate coordinate in View {v + 1}")
            seen[v].add(tuple(xy))
    return {"all_four_views_present": True, "no_missing_coordinates": True,
            "finite_coordinates_inside_bounds": True, "same_count_each_view": len(points),
            "identical_point_ids_and_order": expected, "four_cover_corners_present": True,
            "no_duplicate_coordinates_per_view": True}


def manual_select(points, images, max_width, max_height):
    """Scaled overview plus full-resolution precision window; no automatic matching."""
    if max_width <= 0 or max_height <= 0:
        raise ValueError("Display limits must be positive")
    selected = []
    try:
        for v, image in enumerate(images):
            bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            scale = min(max_width / image.width, max_height / image.height, 1)
            w, h = round(image.width * scale), round(image.height * scale)
            sx, sy = w / image.width, h / image.height
            overview = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
            clicks, candidate = [], [None]
            title = f"View {v + 1}: overview (click to inspect)"
            detail_title = "Full-resolution detail: click, then Enter to accept"
            crop_origin = [0, 0]
            detail_image = [None]

            def overview_click(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    # Map actual resized pixel centers back to original pixels.
                    fx, fy = (x + .5) / sx - .5, (y + .5) / sy - .5
                    ox = max(0, min(image.width - 400, round(fx) - 200))
                    oy = max(0, min(image.height - 400, round(fy) - 200))
                    crop_origin[:] = [ox, oy]
                    detail_image[0] = bgr[oy:oy + 400, ox:ox + 400].copy()
                    candidate[0] = None

            def detail_click(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN and detail_image[0] is not None:
                    candidate[0] = [float(crop_origin[0] + x), float(crop_origin[1] + y)]

            cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow(detail_title, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(title, overview_click)
            cv2.setMouseCallback(detail_title, detail_click)
            while len(clicks) < len(points):
                p = points[len(clicks)]
                frame = overview.copy()
                for q, (x, y) in zip(points, clicks):
                    cv2.circle(frame, (round(x * sx), round(y * sy)), 4, (0, 0, 255), 1)
                    cv2.putText(frame, q["point_id"], (round(x * sx) + 6, round(y * sy)),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
                cv2.rectangle(frame, (0, 0), (w, 100), (255, 255, 255), -1)
                for line, text in enumerate(textwrap.wrap(f"{p['point_id']}: {p['description']}", width=75)):
                    cv2.putText(frame, text, (10, 25 + line * 22),
                                cv2.FONT_HERSHEY_SIMPLEX, .46, (0, 0, 0), 1)
                cv2.putText(frame, "Click overview, click detail; Enter accept; U undo; Esc cancel",
                            (10, 85), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 0, 0), 1)
                cv2.imshow(title, frame)
                if detail_image[0] is not None:
                    detail = detail_image[0].copy()
                    if candidate[0] is not None:
                        x, y = candidate[0]
                        cv2.circle(detail, (round(x - crop_origin[0]), round(y - crop_origin[1])),
                                   5, (0, 0, 255), 1)
                    cv2.imshow(detail_title, detail)
                key = cv2.waitKey(30) & 255
                if key == 27:
                    raise RuntimeError("Selection cancelled; previous saved coordinates retained")
                if key in (ord("u"), ord("U")) and clicks:
                    clicks.pop()
                    candidate[0] = None
                if key in (10, 13) and candidate[0] is not None:
                    clicks.append(candidate[0][:])
                    candidate[0] = None
                    detail_image[0] = None
            selected.append(clicks)
            cv2.destroyAllWindows()
    finally:
        cv2.destroyAllWindows()
    return [dict(point_id=p["point_id"], description=p["description"],
                 views=[selected[v][i] for v in range(4)]) for i, p in enumerate(points)]


def annotate(image, points, view):
    result = image.copy()
    d = ImageDraw.Draw(result)
    for p in points:
        x, y = p["views"][view]
        color = (255, 0, 0)
        # Open ring centered on the approved coordinate; white halo adds contrast.
        d.ellipse((x - 28, y - 28, x + 28, y + 28), outline="white", width=14)
        d.ellipse((x - 24, y - 24, x + 24, y + 24), outline=color, width=8)
        label_font = font(200, bold=True)
        bounds = d.textbbox((0, 0), p["point_id"], font=label_font, stroke_width=8)
        dx, dy = (-320, -240) if p["point_id"] == "P7" and view < 3 else (55, -150)
        tx = min(max(x + dx, 12), image.width - bounds[2] - 12)
        ty = min(max(y + dy, 12), image.height - bounds[3] - 12)
        d.line((x + 24, y - 24, tx, ty + 85), fill="white", width=14)
        d.line((x + 24, y - 24, tx, ty + 85), fill=color, width=6)
        d.text((tx, ty), p["point_id"], font=label_font, fill=color,
               stroke_width=8, stroke_fill="white")
    return result


def export(points, images):
    output = BASE / "output"
    output.mkdir(exist_ok=True)
    fields = ["point_id", "description"] + [f"view{v}_{axis}" for v in range(1, 5) for axis in ("x", "y")]
    with (output / "part_b_correspondences.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for p in points:
            writer.writerow([p["point_id"], p["description"]] + [n for xy in p["views"] for n in xy])
    # Validate the serialized CSV, not just the in-memory selection.
    with (output / "part_b_correspondences.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    reread = [dict(point_id=r["point_id"], description=r["description"],
                   views=[[float(r[f"view{v}_x"]), float(r[f"view{v}_y"])] for v in range(1, 5)]) for r in rows]
    validate(reread, images)
    render_visualizations(points, images)


def render_visualizations(points, images):
    """Write only the six correspondence PNGs; do not touch data or working images."""
    output = BASE / "output"
    sheet = Image.new("RGB", (1800, 2560), "white")
    closeups = Image.new("RGB", (1280, len(points) * 340 + 140), "white")
    sd, cd = ImageDraw.Draw(sheet), ImageDraw.Draw(closeups)
    cd.text((20, 12), "Corresponding landmarks — full-resolution crops", fill="black", font=font(40, bold=True))
    for v in range(4):
        cd.text((v*320+40, 75), f"View {v+1}", fill="black", font=font(36, bold=True))
    for v, image in enumerate(images):
        marked = annotate(image, points, v)
        marked.save(output / f"view_{v + 1}_correspondences.png")
        preview = marked.copy()
        preview.thumbnail((880, 1190), Image.Resampling.LANCZOS)
        ox, oy = (v % 2) * 900, (v // 2) * 1280
        sd.text((ox + 15, oy + 5), f"View {v + 1} — {NAMES[v]}", fill="black", font=font(54, bold=True))
        sheet.paste(preview, (ox + (900 - preview.width) // 2, oy + 70))
        for i, p in enumerate(points):
            x, y = p["views"][v]
            left = max(0, min(image.width - 240, round(x) - 120))
            top = max(0, min(image.height - 240, round(y) - 120))
            crop = image.crop((left, top, left + 240, top + 240))
            draw = ImageDraw.Draw(crop)
            draw.ellipse((x-left-10, y-top-10, x-left+10, y-top+10), outline="white", width=6)
            draw.ellipse((x-left-8, y-top-8, x-left+8, y-top+8), outline=(255, 0, 0), width=3)
            row_y = 140 + i*340
            cd.text((v*320+40, row_y), p['point_id'], fill=(255, 0, 0), font=font(36, bold=True))
            cd.text((v*320+40, row_y+40), f"({x:g}, {y:g})", fill="black", font=font(30, bold=True))
            closeups.paste(crop, (v*320+40, row_y+80))
    sheet.save(output / "part_b_correspondence_comparison.png")
    closeups.save(output / "part_b_landmark_closeups.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual", action="store_true", help="Reselect the same IDs across Views 1–4")
    parser.add_argument("--visuals-only", action="store_true", help="Redraw PNGs from the approved CSV without preparing images or writing numerical data")
    parser.add_argument("--max-width", type=int, default=850)
    parser.add_argument("--max-height", type=int, default=1000)
    args = parser.parse_args()
    if args.visuals_only:
        if args.manual:
            parser.error("--visuals-only cannot be combined with --manual")
        with (BASE / "output" / "part_b_correspondences.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        points = [dict(point_id=r["point_id"], description=r["description"],
                       views=[[float(r[f"view{v}_x"]), float(r[f"view{v}_y"])] for v in range(1, 5)]) for r in rows]
        images = []
        for v in range(1, 5):
            with Image.open(BASE / "working" / f"view_{v}_upright.png") as source:
                images.append(source.convert("RGB"))
        validate(points, images)
        render_visualizations(points, images)
        print("Redrew only four annotated views, comparison, and close-ups from approved coordinates.")
        return
    before = hashes()
    images, metadata = prepare()
    selection_path = BASE / "correspondence_selection.json"
    selection = json.loads(selection_path.read_text())
    points = selection["points"]
    if args.manual:
        points = manual_select(points, images, args.max_width, args.max_height)
    checks = validate(points, images)
    if args.manual:
        selection["points"] = points
        selection["method"] = "Human clicks on full-resolution detail crops"
        selection_path.write_text(json.dumps(selection, indent=2) + "\n")
    export(points, images)
    after = hashes()
    if before != after:
        raise RuntimeError("An original JPEG hash changed")
    report = dict(coordinate_convention=CONVENTION, views=metadata,
                  point_count=len(points), checks=checks, original_sha256=before,
                  original_hashes_unchanged=True, method=selection["method"],
                  book_dimensions_cm=selection["book_dimensions_cm"],
                  rejected_candidates=selection["rejected_candidates"],
                  accuracy_note=selection["accuracy_note"])
    (BASE / "output" / "part_b_correspondence_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
