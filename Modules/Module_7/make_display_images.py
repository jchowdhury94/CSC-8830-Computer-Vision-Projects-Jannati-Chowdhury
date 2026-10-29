"""Write optimized display copies of the notebook photographs for the Streamlit app.

    python make_display_images.py

Reads data/object_measurement/{left,right}/ (never modified) and writes display/pair_N_{left,right}.jpg, EXIF-oriented and
downscaled uniformly to DISPLAY_WIDTH x DISPLAY_HEIGHT. The app maps pixel coordinates with the actual dimensions of these files,
so they are for display only; no measurement ever uses them.
"""
from pathlib import Path

from PIL import Image, ImageOps

from stereo_measure.config import CALIB_PHOTO_DIR, IMAGE_SIZE, NOTEBOOK_PAIR_IDS, NOTEBOOK_PHOTO_DIR, ROOT

DISPLAY_DIR = ROOT / "display"
DISPLAY_WIDTH = 1200
DISPLAY_HEIGHT = round(DISPLAY_WIDTH * IMAGE_SIZE[1] / IMAGE_SIZE[0])


# Sample checkerboard for the calibration reference: left (iPhone Air) calibration photo 8, chosen by inspection (whole board and
# paper visible, 2.0 deg tilt from front-on, sharp). Cropped (x0, y0, w, h in the 4284 x 5712 frame) around the board, then shown
# at 800 px wide; the 48 detected corners lie well inside the crop.
CALIB_SAMPLE = dict(side="left", prefix="air", pair=8, crop=(1900, 3000, 1600, 1800), width=800)


def make_calibration_sample():
    c = CALIB_SAMPLE
    src = CALIB_PHOTO_DIR / c["side"] / f"{c['prefix']}_{c['pair']}.jpeg"
    im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    assert im.size == IMAGE_SIZE, (src, im.size)
    x0, y0, w, h = c["crop"]
    out = DISPLAY_DIR / f"calibration_sample_{c['side']}_pair_{c['pair']}.jpg"
    im.crop((x0, y0, x0 + w, y0 + h)).resize((c["width"], round(c["width"] * h / w)), Image.LANCZOS).save(out, quality=88, optimize=True)
    print(f"{out.relative_to(ROOT)}  {out.stat().st_size / 1024:.0f} KB")


def main():
    DISPLAY_DIR.mkdir(exist_ok=True)
    for pair in NOTEBOOK_PAIR_IDS:
        for side, prefix in (("left", "air"), ("right", "pro")):
            src = NOTEBOOK_PHOTO_DIR / side / f"{prefix}_image{pair}.jpeg"
            im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
            assert im.size == IMAGE_SIZE, (src, im.size)
            out = DISPLAY_DIR / f"pair_{pair}_{side}.jpg"
            im.resize((DISPLAY_WIDTH, DISPLAY_HEIGHT), Image.LANCZOS).save(out, quality=85, optimize=True)
            print(f"{out.relative_to(ROOT)}  {out.stat().st_size / 1024:.0f} KB")
    make_calibration_sample()


if __name__ == "__main__":
    main()
