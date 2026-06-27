"""Offline test harness: run the QR pipeline over saved images and videos.

Burada pipeline'ı kopyalamıyoruz. scanqrcode.py dosyasındaki kodu test ediyoruz. 
(scanqrcode.py'deki main() if __name__ korumasu altında olduğu için import sırasında kamera açılmaz.
Not: Bu yüzden bu koruma önemliydi.)
"""

import time
from pathlib import Path

import cv2

from scanqrcode import build_detector, decode_frame, draw_quad, build_locator

BASE = Path(__file__).resolve().parent
IMAGE_DIR = BASE / "data" / "test_images"
VIDEO_DIR = BASE / "data" / "test_videos"
RESULTS_DIR = BASE / "data" / "results"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

def test_images(detector, locator):
    files = sorted(p for p in IMAGE_DIR.glob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        print(f"[images] none found in {IMAGE_DIR}")
        return
    out_dir = RESULTS_DIR / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[images] {len(files)} file(s)")
    read = 0
    for path in files:
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"  ????  {path.name}: cannot be read file")
            continue
        results = decode_frame(detector, locator, frame)
        if results:
            read += 1
            print(f"  OK    {path.name}: {results[0][0]}")
        else:
            print(f"  FAIL   {path.name}")
        cv2.imwrite(str(out_dir / path.name), annotate(frame, results))
    print(f"[images] read {read}/{len(files)} -> {out_dir}\n")

def test_videos(detector, locator):
    files = sorted(p for p in VIDEO_DIR.glob("*") if p.suffix.lower() in VIDEO_EXTS)
    if not files:
        print(f"[videos] none found in {VIDEO_DIR}")
        return
    out_dir = RESULTS_DIR / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[videos] {len(files)} file(s)")
    for path in files:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            print(f"  {path.name}: cannot open")
            continue

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = None # Lazily created from the first real frame

        total = decoded = 0
        start = time.perf_counter()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            total += 1
            results = decode_frame(detector, locator, frame)
            if results:
                decoded += 1
            out = annotate(frame, results)

            if writer is None:
                h, w = out.shape[:2]
                writer = cv2.VideoWriter(
                    str(out_dir / f"{path.stem}_annotated.avi"),
                    fourcc, src_fps, (w, h),
                )
                if not writer.isOpened():
                    print(f"  {path.name}: VideoWriter failed to open")
                    break
            writer.write(out)
        elapsed = time.perf_counter() - start

        cap.release()
        if writer is not None:
            writer.release()
        fps = total / elapsed if elapsed > 0 else 0.0
        rate = 100 * decoded / total if total else 0.0
        print(f"  {path.name}: read {decoded}/{total} ({rate:.1f}%), "
              f"pipeline {fps:.1f} FPS")
    print(f"[videos] outputs -> {out_dir}\n")

def annotate(frame, results):
    """Draw red quads for every detected QR and a status label on a copy."""
    out = frame.copy()
    for text, points in results:
        draw_quad(out, points)
    label = results[0][0] if results else "NO QR"
    color = (0, 200, 0) if results else (0, 0, 255) # green = read, red = fail
    cv2.putText(out, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2, cv2.LINE_AA)
    return out

def main():
    detector = build_detector()
    locator = build_locator()
    test_images(detector, locator)
    test_videos(detector, locator)

if __name__ == "__main__":
    main()
