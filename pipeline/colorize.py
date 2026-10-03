"""
colorize.py
-----------
Attach RGB colour to the fused point cloud by sampling the aligned walkthrough
video (`rgb.mp4`).

The capture stores one RGB video frame per odometry pose / depth frame (same
count, same order). So for a chosen frame we can:
  1. back-project its depth to 3-D camera points (as in pointcloud.py),
  2. read the matching RGB video frame,
  3. look up each point's colour at its (u, v) pixel,
  4. transform the coloured points to world coordinates.

This produces a colour point cloud, which is a better artefact for a human to
inspect than a bare grey one and demonstrates RGB-plus-geometry fusion. It is
NOT a "video tier" (that would reconstruct geometry from video alone, which
needs monocular SLAM and is out of scope); here the depth still comes from
LiDAR. The distinction is stated plainly.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from .scan_reader import Scan, DEPTH_W, DEPTH_H
from .pointcloud import MIN_DEPTH_M, MAX_DEPTH_M, MIN_CONFIDENCE, VOXEL_SIZE_M, _pixel_grid


def build_colored_point_cloud(
    scan: Scan,
    frame_stride: int = 10,
    voxel_size: float = VOXEL_SIZE_M,
) -> o3d.geometry.PointCloud:
    """Fuse selected frames into a colour point cloud (geometry from LiDAR,
    colour sampled from the aligned RGB video)."""
    video_path = scan.path / "rgb.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"No rgb.mp4 in {scan.path}")

    fx = scan.intrinsics[0, 0]
    fy = scan.intrinsics[1, 1]
    cx = scan.intrinsics[0, 2]
    cy = scan.intrinsics[1, 2]
    x_dir, y_dir = _pixel_grid(fx, fy, cx, cy)

    # Pixel (u, v) grids at depth resolution, used to index the RGB frame
    # (after scaling it down to the depth resolution so indices line up).
    uu, vv = np.meshgrid(np.arange(DEPTH_W), np.arange(DEPTH_H))

    cap = cv2.VideoCapture(str(video_path))
    want = set(f.index for f in scan.frames[::frame_stride])
    frames_by_index = {f.index: f for f in scan.frames}

    pts_all: list[np.ndarray] = []
    col_all: list[np.ndarray] = []

    while True:
        # Read the TRUE frame index from the decoder before each frame, so a
        # mid-stream decode gap can never desync colour from pose (the video
        # may have one fewer decodable frame than the odometry count).
        pos = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
        ok, bgr = cap.read()
        if not ok:
            break
        if pos in want and pos in frames_by_index:
            frame = frames_by_index[pos]
            depth = frame.load_depth_m()
            conf = frame.load_confidence()
            mask = (depth > MIN_DEPTH_M) & (depth < MAX_DEPTH_M) & (conf >= MIN_CONFIDENCE)
            if mask.any():
                d = depth[mask]
                cam = np.stack([x_dir[mask] * d, y_dir[mask] * d, d], axis=1)
                world = cam @ frame.pose[:3, :3].T + frame.pose[:3, 3]

                # Downsize RGB to depth resolution so pixel indices match.
                small = cv2.resize(bgr, (DEPTH_W, DEPTH_H), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                cols = rgb[vv[mask], uu[mask]].astype(np.float32) / 255.0

                pts_all.append(world.astype(np.float32))
                col_all.append(cols)

    cap.release()
    if not pts_all:
        raise ValueError("No coloured points produced")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.concatenate(pts_all))
    pcd.colors = o3d.utility.Vector3dVector(np.concatenate(col_all))
    if voxel_size and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return pcd


if __name__ == "__main__":
    import sys
    from .scan_reader import load_scan

    target = sys.argv[1] if len(sys.argv) > 1 else "sample_data/single_room/c00a170fe1"
    scan = load_scan(target)
    print(f"Colorizing '{scan.name}' ...")
    pcd = build_colored_point_cloud(scan, frame_stride=10)
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    out = Path("outputs") / scan.name / "cloud_colored.ply"
    out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out), pcd)
    print(f"{len(pts):,} coloured points -> {out}")
    print(f"mean RGB: {cols.mean(axis=0).round(2)}")
