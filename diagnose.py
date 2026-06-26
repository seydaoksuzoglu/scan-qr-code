"""Diagnostic: where does the pipeline die on a given video?

For each frame it reports whether the QR was LOCATED (QRCodeDetector.detect),
whether it was large enough, and whether it decoded after warping. This tells
us if the bottleneck is LOCATION or DECODING.
"""

import sys
from pathlib import Path

import cv2

from scanqrcode import (
    build_detector, find_qr_corners, warp_qr, decode_image, quad_side, MIN_QR_SIDE,
)

BASE = Path(__file__).resolve().parent
VIDEO_DIR = BASE / "data" / "test_videos"


def diagnose(name):
    detector = build_detector()
    locator = cv2.QRCodeDetector()
    cap = cv2.VideoCapture(str(VIDEO_DIR / name))
    if not cap.isOpened():
        print(f"cannot open {name}")
        return

    total = located = located_big = decoded = 0
    max_side = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        total += 1
        corners = find_qr_corners(locator, frame)
        if corners is None:
            continue
        located += 1
        side = quad_side(corners)
        max_side = max(max_side, side)
        if side >= MIN_QR_SIDE:
            located_big += 1
        if decode_image(detector, warp_qr(frame, corners)):
            decoded += 1
    cap.release()

    print(f"{name}:")
    print(f"  total frames     : {total}")
    print(f"  QR located       : {located} ({100*located/total:.1f}%)")
    print(f"  located >= {MIN_QR_SIDE}px : {located_big}")
    print(f"  decoded via warp : {decoded}")
    print(f"  largest located  : {max_side:.0f}px")


if __name__ == "__main__":
    targets = sys.argv[1:] or ["go_qr_1.mp4", "go_qr_2.mp4", "nano_qr_video_1.mp4", "qr_deneme_1.mp4", " qr_deneme_2.mp4", "qr_deneme_3.mp4", "qr_deneme_4.mp4", "video_test.mp4", ]
    for t in targets:
        diagnose(t)
