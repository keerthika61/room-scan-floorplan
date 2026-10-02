"""
scan_reader.py
--------------
Reads a StrayScanner-style iPhone LiDAR scan folder into memory.

A scan folder looks like:

    <scan>/
      depth/000000.png ...        16-bit depth in millimeters, 256x192
      confidence/000000.png ...   8-bit LiDAR confidence 0/1/2, 256x192
      odometry.csv                per-frame camera pose (world <- camera)
      camera_matrix.csv           3x3 intrinsics at RGB resolution
      rgb.mp4                     walkthrough video (not needed for geometry)
      imu.csv                     raw IMU (not used in the core LiDAR path)

The two things that matter for geometry are:
  1. intrinsics  -- how pixels map to ray directions
  2. per-frame pose -- where the camera was and how it was oriented

This module only *loads and validates*. Turning depth into 3D points
happens in the next stage (point cloud generation).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# Depth / confidence images are captured at this fixed low resolution.
DEPTH_W, DEPTH_H = 256, 192


@dataclass
class Frame:
    """One captured moment: an index, a pose, and file paths to its images."""

    index: int
    # 4x4 transform that maps a point in camera coordinates to world
    # coordinates (camera-to-world). Built from the odometry position +
    # quaternion.
    pose: np.ndarray
    depth_path: Path
    confidence_path: Path

    def load_depth_m(self) -> np.ndarray:
        """Return depth as a (H, W) float array in METERS. 0 = no reading."""
        raw = np.asarray(Image.open(self.depth_path), dtype=np.float32)
        return raw / 1000.0  # millimeters -> meters

    def load_confidence(self) -> np.ndarray:
        """Return confidence as a (H, W) uint8 array with values 0/1/2."""
        return np.asarray(Image.open(self.confidence_path), dtype=np.uint8)


@dataclass
class Scan:
    """A whole scan: intrinsics plus the ordered list of frames."""

    path: Path
    name: str
    # 3x3 intrinsics already SCALED to the depth resolution (256x192).
    intrinsics: np.ndarray
    frames: list[Frame]

    def __len__(self) -> int:
        return len(self.frames)


def _quaternion_to_rotation(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert a unit quaternion to a 3x3 rotation matrix.

    Standard formula; avoids pulling in scipy just for this. The quaternion
    is normalized first so small numeric drift in the logs doesn't matter.
    """
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n == 0:
        return np.eye(3)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def _load_intrinsics_scaled(scan_dir: Path) -> np.ndarray:
    """Load camera_matrix.csv (at RGB resolution) and scale it to depth res.

    The intrinsics in the file correspond to the full-resolution RGB frames
    (1920x1440). Depth is 256x192, so fx, fy, cx, cy must be scaled by the
    ratio of resolutions, otherwise every 3D point would be wrong.
    """
    m = np.loadtxt(scan_dir / "camera_matrix.csv", delimiter=",")
    if m.shape != (3, 3):
        raise ValueError(f"camera_matrix.csv should be 3x3, got {m.shape}")

    rgb_w = 2.0 * m[0, 2]  # cx is ~ width/2, a reliable way to recover width
    rgb_h = 2.0 * m[1, 2]
    sx = DEPTH_W / rgb_w
    sy = DEPTH_H / rgb_h

    k = m.copy()
    k[0, 0] *= sx  # fx
    k[1, 1] *= sy  # fy
    k[0, 2] *= sx  # cx
    k[1, 2] *= sy  # cy
    return k


def load_scan(scan_dir: str | Path) -> Scan:
    """Load a scan folder into a Scan object.

    Raises a clear error if required files are missing so the single-command
    entry point can fail fast with something a human can act on.
    """
    scan_dir = Path(scan_dir)
    if not scan_dir.is_dir():
        raise FileNotFoundError(f"Scan folder not found: {scan_dir}")

    odom_path = scan_dir / "odometry.csv"
    depth_dir = scan_dir / "depth"
    conf_dir = scan_dir / "confidence"
    for required in (odom_path, scan_dir / "camera_matrix.csv", depth_dir, conf_dir):
        if not required.exists():
            raise FileNotFoundError(f"Missing required input: {required}")

    intrinsics = _load_intrinsics_scaled(scan_dir)

    # odometry.csv columns:
    #   timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, ...
    rows = np.loadtxt(odom_path, delimiter=",", skiprows=1, usecols=range(9))

    frames: list[Frame] = []
    for row in rows:
        idx = int(row[1])
        x, y, z = row[2], row[3], row[4]
        qx, qy, qz, qw = row[5], row[6], row[7], row[8]

        pose = np.eye(4)
        pose[:3, :3] = _quaternion_to_rotation(qx, qy, qz, qw)
        pose[:3, 3] = [x, y, z]

        stem = f"{idx:06d}"
        depth_path = depth_dir / f"{stem}.png"
        conf_path = conf_dir / f"{stem}.png"
        if not depth_path.exists() or not conf_path.exists():
            # Odometry sometimes lists a frame whose images weren't written.
            continue

        frames.append(Frame(idx, pose, depth_path, conf_path))

    if not frames:
        raise ValueError(f"No usable frames found in {scan_dir}")

    return Scan(path=scan_dir, name=scan_dir.name, intrinsics=intrinsics, frames=frames)


if __name__ == "__main__":
    # Simple self-test / inspection when run directly.
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "sample_data/single_room/c00a170fe1"
    scan = load_scan(target)
    print(f"Loaded scan '{scan.name}' with {len(scan)} frames")
    print("Intrinsics (scaled to depth 256x192):")
    print(scan.intrinsics)

    f0 = scan.frames[0]
    d = f0.load_depth_m()
    c = f0.load_confidence()
    valid = d[d > 0]
    print(f"\nFrame {f0.index}:")
    print(f"  depth shape {d.shape}, {valid.size} valid pixels")
    if valid.size:
        print(f"  depth range {valid.min():.2f} m .. {valid.max():.2f} m")
    print(f"  confidence values present: {sorted(np.unique(c).tolist())}")
    print(f"  pose translation: {f0.pose[:3, 3]}")
