"""
pointcloud.py
-------------
Turn a loaded Scan into a single fused 3D point cloud in world coordinates.

The core operation is "back-projection":

    For a pixel (u, v) with depth d, its 3D position in the CAMERA frame is
        X = (u - cx) / fx * d
        Y = (v - cy) / fy * d
        Z = d
    Then the camera-to-world pose places it in the ROOM frame.

We fuse many frames into one cloud. Because a scan has thousands of frames
and each has up to 49k pixels, we:
  - keep only high-confidence depth pixels,
  - drop pixels that are too close/far (noise, or seeing through doorways),
  - process every Nth frame (frame_stride),
  - voxel-downsample the final cloud so downstream geometry is fast.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

from .scan_reader import Scan, DEPTH_W, DEPTH_H

# Depth readings outside this range are discarded. LiDAR on these devices is
# unreliable past ~5 m, and sub-0.2 m usually means the sensor saw itself /
# the operator.
MIN_DEPTH_M = 0.2
MAX_DEPTH_M = 5.0

# Keep pixels with confidence >= this. 0=low, 1=medium, 2=high.
MIN_CONFIDENCE = 2

# Final cloud voxel size (meters). 2 cm keeps walls crisp but shrinks the
# cloud to a size plane-fitting can chew through quickly.
VOXEL_SIZE_M = 0.02


def _pixel_grid(fx: float, fy: float, cx: float, cy: float):
    """Precompute the (u-cx)/fx and (v-cy)/fy ray-direction grids once.

    These only depend on intrinsics, so computing them a single time and
    reusing across all frames saves a lot of work.
    """
    us = np.arange(DEPTH_W, dtype=np.float32)
    vs = np.arange(DEPTH_H, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)  # shape (H, W)
    x_dir = (uu - cx) / fx
    y_dir = (vv - cy) / fy
    return x_dir, y_dir


def build_point_cloud(
    scan: Scan,
    frame_stride: int = 5,
    min_confidence: int = MIN_CONFIDENCE,
    voxel_size: float = VOXEL_SIZE_M,
) -> o3d.geometry.PointCloud:
    """Fuse selected frames of a scan into one world-space point cloud."""
    fx = scan.intrinsics[0, 0]
    fy = scan.intrinsics[1, 1]
    cx = scan.intrinsics[0, 2]
    cy = scan.intrinsics[1, 2]
    x_dir, y_dir = _pixel_grid(fx, fy, cx, cy)

    chunks: list[np.ndarray] = []

    for frame in scan.frames[::frame_stride]:
        depth = frame.load_depth_m()
        conf = frame.load_confidence()

        mask = (
            (depth > MIN_DEPTH_M)
            & (depth < MAX_DEPTH_M)
            & (conf >= min_confidence)
        )
        if not mask.any():
            continue

        d = depth[mask]
        # Points in the camera frame.
        xc = x_dir[mask] * d
        yc = y_dir[mask] * d
        zc = d
        cam_pts = np.stack([xc, yc, zc], axis=1)  # (N, 3)

        # Transform to world: p_world = R @ p_cam + t
        pose = frame.pose
        world = cam_pts @ pose[:3, :3].T + pose[:3, 3]
        chunks.append(world.astype(np.float32))

    if not chunks:
        raise ValueError("No points survived filtering; try lowering min_confidence.")

    all_pts = np.concatenate(chunks, axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)
    if voxel_size and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)

    # Remove sparse outliers (stray points from reflections / depth noise).
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return pcd


def save_point_cloud(pcd: o3d.geometry.PointCloud, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out_path), pcd)
    return out_path


if __name__ == "__main__":
    import sys
    from .scan_reader import load_scan

    target = sys.argv[1] if len(sys.argv) > 1 else "sample_data/single_room/c00a170fe1"
    scan = load_scan(target)
    print(f"Loaded '{scan.name}' with {len(scan)} frames; building point cloud...")

    pcd = build_point_cloud(scan)
    pts = np.asarray(pcd.points)
    print(f"Fused cloud: {len(pts):,} points")
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    print(f"  bounds X: {mn[0]:.2f} .. {mx[0]:.2f} m  (extent {mx[0]-mn[0]:.2f})")
    print(f"  bounds Y: {mn[1]:.2f} .. {mx[1]:.2f} m  (extent {mx[1]-mn[1]:.2f})")
    print(f"  bounds Z: {mn[2]:.2f} .. {mx[2]:.2f} m  (extent {mx[2]-mn[2]:.2f})")

    out = save_point_cloud(pcd, f"outputs/{scan.name}/cloud.ply")
    print(f"Saved -> {out}")
