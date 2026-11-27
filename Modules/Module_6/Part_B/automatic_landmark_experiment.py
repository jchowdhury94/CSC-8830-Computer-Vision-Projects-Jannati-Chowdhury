"""Part B book experiment: correspondences and structure-from-motion reconstruction (shared sfm_pipeline functions).

Method
1. P1-P4 in View 1 are the verified reference corners. Nothing else is given to the correspondence method.
2. sfm_pipeline.find_correspondences (the visitor page's function) identifies P5-P8 in all four views and
   estimates P1-P4 in Views 2-4.
3. The verified P1-P4 coordinates of Views 2-4 replace the corner estimates (explicit corrections).
4. P5-P8 keep their automatic coordinates unless listed in LANDMARK_CORRECTIONS after visual verification.
5. sfm_pipeline.reconstruct factorizes the final coordinates, aligns and scales the points and measures the edges.

Inputs (read only)
    correspondence_selection.json   verified P1-P4 corners, measured cover width and height
    submission_images/*.jpeg        the four photographs (full resolution, no location metadata)

Run from any directory:
    .venv/bin/python Modules/Module_6/Part_B/automatic_landmark_experiment.py
    .venv/bin/python Modules/Module_6/Part_B/automatic_landmark_experiment.py --figures-only

Requires NumPy, OpenCV and Pillow (repository requirements.txt).

Outputs (only output/revised/ is written)
    part_b_correspondences.csv        final P1-P8 coordinates; the input of the results page
    part_b_automatic_points.csv       automatic positions, final positions and their source
    part_b_reconstructed_points_cm.csv, part_b_measurements.csv, part_b_summary.json
    part_b_correspondence_views.png, part_b_landmark_closeups.png, part_b_selection_example.jpg
--figures-only redraws the two results-page figures from the saved coordinates.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

import sfm_pipeline as sfm
import visitor_reconstruction as vr

BASE = Path(__file__).resolve().parent
PHOTOS = BASE / "submission_images"
OUT = BASE / "output" / "revised"
NAMES = ("IMG_7596.jpeg", "IMG_7597.jpeg", "IMG_7598.jpeg", "IMG_7603.jpeg")
# Manual corrections of automatic landmarks after visual verification: {(view, "P5"): (x, y)} in original pixels.
LANDMARK_CORRECTIONS = {}
CROP = 240


def load_selection():
    """-> the selection record and the verified (4, 4, 2) P1-P4 coordinates (view, corner, x/y)."""
    selection = json.loads((BASE / "correspondence_selection.json").read_text(encoding="utf-8"))
    if tuple(p["point_id"] for p in selection["points"]) != sfm.POINT_IDS[:4]:
        raise ValueError("Expected the corners P1-P4 in order")
    corners = np.array([p["views"] for p in selection["points"]], dtype=float).transpose(1, 0, 2)
    return selection, corners


def final_coordinates(automatic, corners):
    """Final (4, 8, 2) coordinates and their origin: verified corners, automatic landmarks, listed corrections."""
    final = np.round(automatic, 2)
    final[:, :4] = corners
    source = [["verified corner"] * 4 + ["automatic"] * 4 for _ in range(sfm.VIEW_COUNT)]
    for (view, point_id), xy in LANDMARK_CORRECTIONS.items():
        i = sfm.POINT_IDS.index(point_id)
        final[view - 1, i] = xy
        source[view - 1][i] = "corrected manually"
    if not np.isfinite(final).all():
        raise ValueError("Some landmarks are missing; add them to LANDMARK_CORRECTIONS after selecting them.")
    return final, source


# --------------------------------------------------------------------------- figures

def font(size, bold=False):
    names = ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", "DejaVuSans-Bold.ttf") if bold else (
        "/System/Library/Fonts/Supplemental/Arial.ttf", "DejaVuSans.ttf")
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def annotate(photo, final, view):
    """Copy of one photograph with an open ring and an offset label on every final point."""
    result = photo.copy()
    d = ImageDraw.Draw(result)
    for i, point_id in enumerate(sfm.POINT_IDS):
        x, y = (float(value) for value in final[view, i])
        color = (255, 0, 0)
        # Open ring centered on the coordinate; white halo adds contrast.
        d.ellipse((x - 28, y - 28, x + 28, y + 28), outline="white", width=14)
        d.ellipse((x - 24, y - 24, x + 24, y + 24), outline=color, width=8)
        label_font = font(200, bold=True)
        bounds = d.textbbox((0, 0), point_id, font=label_font, stroke_width=8)
        dx, dy = (-320, -240) if point_id == "P7" and view < 3 else (55, -150)
        tx = min(max(x + dx, 12), photo.width - bounds[2] - 12)
        ty = min(max(y + dy, 12), photo.height - bounds[3] - 12)
        d.line((x + 24, y - 24, tx, ty + 85), fill="white", width=14)
        d.line((x + 24, y - 24, tx, ty + 85), fill=color, width=6)
        d.text((tx, ty), point_id, font=label_font, fill=color, stroke_width=8, stroke_fill="white")
    return result


def draw_ring(draw, xy, radius, color, width):
    x, y = xy
    draw.ellipse((x - radius - 3, y - radius - 3, x + radius + 3, y + radius + 3), outline="white", width=width + 6)
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=width)


def views_sheet(photos, final):
    sheet = Image.new("RGB", (1800, 2560), "white")
    d = ImageDraw.Draw(sheet)
    for v, photo in enumerate(photos):
        marked = annotate(photo, final, v)
        marked.thumbnail((880, 1190), Image.Resampling.LANCZOS)
        ox, oy = (v % 2) * 900, (v // 2) * 1280
        d.text((ox + 15, oy + 5), f"View {v + 1}", fill="black", font=font(54, bold=True))
        sheet.paste(marked, (ox + (900 - marked.width) // 2, oy + 70))
    return sheet


def landmark_closeups(photos, final):
    """P1-P8 x Views 1-4: unscaled full-resolution crops; the ring is open so the feature stays visible."""
    sheet = Image.new("RGB", (4 * 320, sfm.POINT_COUNT * 340 + 140), "white")
    d = ImageDraw.Draw(sheet)
    d.text((20, 12), "Corresponding landmarks — full-resolution crops", fill="black", font=font(40, bold=True))
    for v, photo in enumerate(photos):
        d.text((v * 320 + 40, 75), f"View {v + 1}", fill="black", font=font(36, bold=True))
        for i, name in enumerate(sfm.POINT_IDS):
            center = final[v, i]
            left = int(np.clip(round(center[0]) - CROP // 2, 0, photo.width - CROP))
            top = int(np.clip(round(center[1]) - CROP // 2, 0, photo.height - CROP))
            crop = photo.crop((left, top, left + CROP, top + CROP))
            draw_ring(ImageDraw.Draw(crop), center - (left, top), 8, (255, 0, 0), 3)
            x, y = v * 320 + 40, 140 + i * 340
            d.text((x, y), name, fill=(255, 0, 0), font=font(36, bold=True))
            d.text((x, y + 40), f"({center[0]:.2f}, {center[1]:.2f})", fill="black", font=font(28, bold=True))
            sheet.paste(crop, (x, y + 80))
    return sheet


def upright_photos():
    """The four photographs at full resolution with EXIF orientation applied."""
    photos = []
    for name in NAMES:
        with Image.open(PHOTOS / name) as source:
            photos.append(ImageOps.exif_transpose(source).convert("RGB"))
    return photos


def render_figures(photos, final):
    views_sheet(photos, final).save(OUT / "part_b_correspondence_views.png")
    landmark_closeups(photos, final).save(OUT / "part_b_landmark_closeups.png")


def selection_example(decoded, final):
    """The visitor page's example figure: the final points on View 1."""
    image = decoded[0]
    statuses = ["selected"] * 4 + ["automatic"] * 4
    marked = vr.render_points(image.rgb, sfm.to_working(final[0], image), statuses, radius=14, font_scale=1.4)
    example = Image.fromarray(marked)
    example.thumbnail((900, 900), Image.Resampling.LANCZOS)
    return example


# --------------------------------------------------------------------------- data

def saved_coordinates():
    """The saved final (4, 8, 2) coordinates, exactly as reconstructed."""
    with (OUT / "part_b_correspondences.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if tuple(r["point_id"] for r in rows) != sfm.POINT_IDS:
        raise ValueError("Unexpected correspondence order")
    return np.array([[[float(r[f"view{v}_x"]), float(r[f"view{v}_y"])] for r in rows] for v in range(1, 5)])


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def fmt(value, places=2):
    return "" if value is None or not np.isfinite(value) else f"{value:.{places}f}"


def digests():
    return {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(BASE.rglob("*"))
            if p.is_file() and OUT not in p.parents and "__pycache__" not in p.parts}


def main():
    before = digests()
    selection, corners = load_selection()
    decoded = [vr.decode_image((PHOTOS / name).read_bytes(), name) for name in NAMES]
    if any(image.original_size != (4284, 5712) for image in decoded):
        raise ValueError("Unexpected upright image size")
    # Only the four View 1 corners are given to the method.
    found = sfm.find_correspondences(decoded, corners[0])
    final, source = final_coordinates(found.points, corners)
    width, height = selection["book_dimensions_cm"]["width"], selection["book_dimensions_cm"]["height"]
    result = sfm.reconstruct(final, width, height)
    measurements = [dict(edge=m["label"], measured_cm=m["actual"], reconstructed_cm=m["reconstructed"],
                         error_percent=m["error"]) for m in result.measurements]

    OUT.mkdir(parents=True, exist_ok=True)
    columns = [f"view{v}_{a}" for v in range(1, 5) for a in ("x", "y")]
    write_csv(OUT / "part_b_correspondences.csv", ["point_id", *columns],
              ([name, *(f"{value:.2f}" for value in final[:, i].ravel())] for i, name in enumerate(sfm.POINT_IDS)))
    write_csv(OUT / "part_b_automatic_points.csv",
              ("view", "point_id", "automatic_x", "automatic_y", "final_x", "final_y", "difference_px", "final_source"),
              ([v + 1, name, fmt(found.points[v, i, 0]), fmt(found.points[v, i, 1]), fmt(final[v, i, 0]),
                fmt(final[v, i, 1]), fmt(float(np.linalg.norm(found.points[v, i] - final[v, i]))), source[v][i]]
               for v in range(4) for i, name in enumerate(sfm.POINT_IDS)))
    write_csv(OUT / "part_b_reconstructed_points_cm.csv", ("point_id", "X_cm", "Y_cm", "Z_cm"),
              ([name, *(repr(float(value)) for value in result.cm[i])] for i, name in enumerate(sfm.POINT_IDS)))
    write_csv(OUT / "part_b_measurements.csv", list(measurements[0]), ([m[k] for k in m] for m in measurements))
    render_figures(upright_photos(), final)
    selection_example(decoded, final).save(OUT / "part_b_selection_example.jpg", quality=90)
    summary = dict(
        status="Correspondences approved; used on the results page",
        landmarks_found=found.landmarks_found,
        views=[dict(view=p.view, matches=p.matches, inliers=p.inliers, coverage=p.coverage) for p in found.views],
        landmark_corrections={f"V{v} {p}": list(xy) for (v, p), xy in LANDMARK_CORRECTIONS.items()},
        singular_values=result.singular_values.tolist(), L_eigenvalues=result.metric["eigenvalues"].tolist(),
        scale_cm_per_unit=result.scale, view_rms_px=result.view_rms.tolist(),
        max_plane_displacement_cm=float(np.abs(result.cm[:, 2] - result.frame["plane_z"] * result.scale).max()),
        measurements=measurements)
    (OUT / "part_b_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if digests() != before:
        raise RuntimeError("A protected Part B file changed")
    print(json.dumps(summary, indent=2))
    for name, i in zip(sfm.POINT_IDS[4:], range(4, 8)):
        print(name, np.round(final[:, i], 2).tolist())


if __name__ == "__main__":
    render_figures(upright_photos(), saved_coordinates()) if "--figures-only" in sys.argv else main()
