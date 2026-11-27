"""Make submission copies of the four photographs without location metadata.

The compressed image data is copied byte for byte (no decoding or recompression).
Only metadata changes: the EXIF GPS directory and the opaque MakerNote are erased,
and the XMP packet is dropped. Camera, exposure, orientation and color-profile
metadata are kept. The files in input/ are only read.

Run from any directory:
    .venv/bin/python Modules/Module_6/Part_B/sanitize_submission_images.py

Inputs: input/IMG_*.jpeg (original photographs, kept locally, not in Git).
Outputs: submission_images/IMG_*.jpeg. Requires NumPy and Pillow (repository requirements.txt).
"""
from pathlib import Path
import hashlib
import io
import struct

import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "input"
DESTINATION = BASE / "submission_images"
NAMES = ("IMG_7596.jpeg", "IMG_7597.jpeg", "IMG_7598.jpeg", "IMG_7603.jpeg")
EXIF_IFD, GPS_IFD, MAKER_NOTE = 0x8769, 0x8825, 0x927C
TYPE_BYTES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 13: 4}


def entries(tiff, order, offset):
    """-> list of (entry position, tag, type, count, value-or-offset) of one TIFF directory."""
    count = struct.unpack_from(order + "H", tiff, offset)[0]
    return [(offset + 2 + 12 * i, *struct.unpack_from(order + "HHII", tiff, offset + 2 + 12 * i)) for i in range(count)]


def erase_value(tiff, kind, count, value):
    size = TYPE_BYTES[kind] * count
    if size > 4:
        tiff[value:value + size] = bytes(size)


def remove_entry(tiff, order, directory, position):
    """Delete one 12-byte entry in place; every other offset in the file stays valid."""
    count = struct.unpack_from(order + "H", tiff, directory)[0]
    end = directory + 2 + 12 * count + 4  # includes the next-directory pointer
    tiff[position:end] = tiff[position + 12:end] + bytes(12)
    struct.pack_into(order + "H", tiff, directory, count - 1)


def clean_tiff(data):
    tiff = bytearray(data)
    order = {b"MM": ">", b"II": "<"}[bytes(tiff[:2])]
    root = struct.unpack_from(order + "I", tiff, 4)[0]
    for position, tag, _, _, value in entries(tiff, order, root):
        if tag == EXIF_IFD:
            for inner_position, inner_tag, kind, count, inner_value in entries(tiff, order, value):
                if inner_tag == MAKER_NOTE:
                    erase_value(tiff, kind, count, inner_value)
                    remove_entry(tiff, order, value, inner_position)
                    break
    for position, tag, _, _, value in entries(tiff, order, root):
        if tag == GPS_IFD:
            gps = entries(tiff, order, value)
            for _, _, kind, count, gps_value in gps:
                erase_value(tiff, kind, count, gps_value)
            size = 2 + 12 * len(gps) + 4
            tiff[value:value + size] = bytes(size)
            remove_entry(tiff, order, root, position)
            break
    return bytes(tiff)


def sanitize(data):
    if data[:2] != b"\xff\xd8":
        raise ValueError("Not a JPEG file")
    output, index = [data[:2]], 2
    while True:
        if data[index] != 0xFF:
            raise ValueError("Unexpected JPEG structure")
        marker = data[index + 1]
        if marker == 0xDA:  # start of scan: the rest is the compressed image, copied unchanged
            output.append(data[index:])
            break
        length = struct.unpack_from(">H", data, index + 2)[0]
        segment = data[index:index + 2 + length]
        payload = segment[4:]
        index += 2 + length
        if marker == 0xE1 and payload.startswith(b"Exif\0\0"):
            segment = segment[:10] + clean_tiff(payload[6:])
        elif marker == 0xE1:  # XMP
            continue
        elif marker == 0xED:  # IPTC / Photoshop block
            continue
        output.append(segment)
    return b"".join(output)


def verify(original, cleaned):
    with Image.open(io.BytesIO(original)) as a, Image.open(io.BytesIO(cleaned)) as b:
        assert a.size == b.size and a.getexif().get(274) == b.getexif().get(274)
        assert a.info.get("icc_profile") == b.info.get("icc_profile")
        assert np.array_equal(np.asarray(a), np.asarray(b)), "pixels differ"
        exif = b.getexif()
        assert GPS_IFD not in exif and not exif.get_ifd(GPS_IFD), "GPS metadata remains"
        assert "xmp" not in b.info
        kept, source = exif.get_ifd(EXIF_IFD), a.getexif().get_ifd(EXIF_IFD)
        assert MAKER_NOTE not in kept
        assert kept == {k: v for k, v in source.items() if k != MAKER_NOTE}, "camera metadata changed"
    assert original[original.index(b"\xff\xda"):][-4096:] == cleaned[-4096:]


def main():
    DESTINATION.mkdir(exist_ok=True)
    for name in NAMES:
        original = (SOURCE / name).read_bytes()
        cleaned = sanitize(original)
        verify(original, cleaned)
        (DESTINATION / name).write_bytes(cleaned)
        assert (SOURCE / name).read_bytes() == original
        print(f"{name}: pixels identical, GPS removed; sha256 {hashlib.sha256(cleaned).hexdigest()}")


if __name__ == "__main__":
    main()
