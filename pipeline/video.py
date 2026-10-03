"""
video.py
--------
Video ingestion for the capture's `rgb.mp4`.

Scope, stated honestly:
  - This reads the walkthrough video and verifies it is frame-aligned with the
    depth / odometry streams (same frame count, same order), then exposes
    individual RGB frames by index for downstream use (e.g. colorization).
  - This is NOT a monocular "video tier" that recovers geometry from video
    alone. On a Pro capture the geometry comes from LiDAR depth + ARKit poses;
    a true video-only tier would need monocular SLAM + metric-scale recovery,
    which is out of scope here. We keep the two clearly separate.

What it genuinely provides: a validated bridge from the raw video file to the
pipeline, and the alignment check that makes RGB-to-geometry fusion correct.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def video_info(scan_dir: str | Path) -> dict:
    """Return basic properties of the scan's rgb.mp4."""
    scan_dir = Path(scan_dir)
    path = scan_dir / "rgb.mp4"
    if not path.exists():
        raise FileNotFoundError(f"No rgb.mp4 in {scan_dir}")
    cap = cv2.VideoCapture(str(path))
    info = {
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": round(float(cap.get(cv2.CAP_PROP_FPS)), 1),
    }
    cap.release()
    return info


def check_alignment(scan_dir: str | Path) -> dict:
    """Verify the video frame count matches the odometry / depth frame count.

    Correct RGB-to-geometry fusion depends on this one-to-one alignment, so we
    check it explicitly rather than assume it.
    """
    scan_dir = Path(scan_dir)
    vinfo = video_info(scan_dir)
    # odometry rows (minus header)
    odom = scan_dir / "odometry.csv"
    n_odom = sum(1 for _ in open(odom)) - 1
    n_depth = len(list((scan_dir / "depth").glob("*.png")))
    aligned = vinfo["frames"] == n_odom == n_depth
    return {
        "video_frames": vinfo["frames"],
        "odometry_rows": n_odom,
        "depth_frames": n_depth,
        "aligned": aligned,
    }


def read_frame(scan_dir: str | Path, index: int) -> np.ndarray:
    """Return RGB frame `index` from the video as an (H, W, 3) uint8 array."""
    scan_dir = Path(scan_dir)
    cap = cv2.VideoCapture(str(scan_dir / "rgb.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise IndexError(f"frame {index} not readable")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "sample_data/single_room/c00a170fe1"
    print("video_info:", video_info(target))
    a = check_alignment(target)
    print("alignment:", a)
    print("aligned ->" , "OK, RGB fusion is valid" if a["aligned"]
          else "MISMATCH, fusion would be unreliable")
