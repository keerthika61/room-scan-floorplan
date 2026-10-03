"""
room_outline.py
---------------
Recover a CLEAN room outline (straight walls) from the point cloud.

The first-pass footprint traced raw floor coverage and produced a jagged
50+ vertex blob. This module fixes that in three steps:

  1. Orientation: indoor walls are mostly parallel/perpendicular to each
     other ("Manhattan"). We estimate the room's dominant wall angle from
     the wall points and rotate the cloud so walls line up with the X/Z
     axes. This lets us snap to clean right angles later.

  2. Segmentation: in the rotated frame `room_segment` isolates the dominant
     enclosed room (breaking narrow doorway necks), or falls back to the full
     footprint when no single compact room dominates.

  3. Outline: we trace the contour of the room mask and simplify it
     (Douglas-Peucker), then drop sub-threshold spurs. The result is a
     simplified free-form polygon whose edges approximate the walls. We do
     NOT force axis-alignment: the sample captures are large, organically
     shaped spaces, so a rectilinear assumption would misrepresent them.
     Orientation is still estimated and removed so the simplification works in
     a consistent frame.

Everything is reported back in the ORIGINAL world frame so measurements and
rendering stay consistent with the rest of the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .room_segment import dominant_room_mask

GRID_RES_M = 0.02
FOOTPRINT_BAND_M = 0.12    # keep points from floor up to here for the footprint
SIMPLIFY_TOL_M = 0.10
MIN_WALL_LEN_M = 0.25      # drop outline edges shorter than this


@dataclass
class RoomOutline:
    polygon_xz: np.ndarray          # (K,2) vertices, world frame
    wall_lengths_m: list[float]
    wall_length_ci_m: list[float]
    floor_area_m2: float
    floor_area_ci_m2: float
    orientation_deg: float          # estimated room rotation that was removed
    grid: np.ndarray                # room mask (rotated frame)
    grid_origin_xz: np.ndarray      # world-ish origin in rotated frame
    rotation: np.ndarray            # 2x2 rot applied (rotated = R @ world_xz)
    grid_res_m: float
    segmentation_mode: str = "dominant_room"  # or "full_footprint_fallback"


def estimate_orientation(points: np.ndarray, floor_y: float) -> float:
    """Estimate the room's dominant orientation (radians, folded to [0, pi/2)).

    The floor footprint is a roughly rectangular blob; the angle of its
    minimum-area bounding rectangle is the room's orientation. This is robust
    and, unlike a Sobel-gradient vote on a binary raster (which locks onto the
    staircase artefacts of a rotated line), it recovers the true angle: on
    synthetic rooms rotated 0-60 deg it is exact to 0 deg (see
    `TestOrientation`).
    """
    band = (points[:, 1] >= floor_y - 0.05) & (
        points[:, 1] <= floor_y + FOOTPRINT_BAND_M
    )
    xz = points[band][:, [0, 2]]
    if xz.shape[0] < 200:
        return 0.0

    mn = xz.min(0) - 0.1
    size = np.ceil((xz.max(0) + 0.1 - mn) / GRID_RES_M).astype(int) + 1
    grid = np.zeros((size[1], size[0]), np.uint8)
    cols = np.clip(((xz[:, 0] - mn[0]) / GRID_RES_M).astype(int), 0, size[0] - 1)
    rows = np.clip(((xz[:, 1] - mn[1]) / GRID_RES_M).astype(int), 0, size[1] - 1)
    grid[rows, cols] = 255

    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    cnt = max(contours, key=cv2.contourArea)
    (_, _), (_, _), angle_deg = cv2.minAreaRect(cnt)  # degrees in (-90, 0]
    return np.radians(angle_deg % 90.0)


def _rotation_matrix(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _outline_polygon(mask: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Trace the room mask contour, convert to world metres, simplify, clean."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    eps = SIMPLIFY_TOL_M / GRID_RES_M
    poly = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2).astype(np.float64)
    poly[:, 0] = origin[0] + poly[:, 0] * GRID_RES_M
    poly[:, 1] = origin[1] + poly[:, 1] * GRID_RES_M
    return _merge_short_edges(poly)


def _merge_short_edges(poly: np.ndarray) -> np.ndarray:
    """Drop vertices that create sub-MIN_WALL_LEN_M edges (removes jitter)."""
    if len(poly) < 4:
        return poly
    cleaned = [poly[0]]
    for p in poly[1:]:
        if np.linalg.norm(p - cleaned[-1]) >= MIN_WALL_LEN_M:
            cleaned.append(p)
    return np.array(cleaned)


def _polygon_area(poly: np.ndarray) -> float:
    x, z = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(z, -1)) - np.dot(z, np.roll(x, -1)))


def compute_room_outline(points: np.ndarray, floor_y: float) -> RoomOutline:
    theta = estimate_orientation(points, floor_y)
    R = _rotation_matrix(-theta)  # rotate world so walls align to axes

    # Build a copy of the cloud rotated into the Manhattan frame (X,Z rotated,
    # Y/height untouched), so segmentation and the outline are axis-aligned.
    pts_rot = points.copy()
    pts_rot[:, [0, 2]] = points[:, [0, 2]] @ R.T

    # Per-room segmentation: isolate the dominant enclosed room, dropping
    # regions seen through doorways. Falls back to the full footprint when no
    # single compact room dominates (and records which happened).
    room, origin, _, seg_info = dominant_room_mask(pts_rot, floor_y,
                                                   footprint_band_m=FOOTPRINT_BAND_M)
    poly_rot = _outline_polygon(room, origin)

    # Rotate polygon back to the original world frame.
    poly_world = poly_rot @ R  # inverse of R.T is R since R orthonormal

    edges = np.roll(poly_world, -1, axis=0) - poly_world
    lengths = np.sqrt((edges**2).sum(axis=1)).tolist()
    cell = GRID_RES_M
    wall_ci = [float(np.sqrt(2) * cell + 0.01 * L) for L in lengths]

    area = _polygon_area(poly_world)
    area_ci = float(sum(lengths) * cell)

    return RoomOutline(
        polygon_xz=poly_world,
        wall_lengths_m=lengths,
        wall_length_ci_m=wall_ci,
        floor_area_m2=area,
        floor_area_ci_m2=area_ci,
        orientation_deg=float(np.degrees(theta)),
        grid=room,
        grid_origin_xz=origin,
        rotation=R,
        grid_res_m=GRID_RES_M,
        segmentation_mode=seg_info["mode"],
    )


if __name__ == "__main__":
    import sys
    import open3d as o3d
    from .planes import find_floor_ceiling

    ply = sys.argv[1] if len(sys.argv) > 1 else "outputs/c00a170fe1/cloud.ply"
    pts = np.asarray(o3d.io.read_point_cloud(ply).points)
    floor = find_floor_ceiling(pts).floor.height
    ro = compute_room_outline(pts, floor)
    print(f"Orientation removed: {ro.orientation_deg:.1f} deg")
    print(f"Room outline: {len(ro.polygon_xz)} vertices (walls)")
    print(f"Floor area: {ro.floor_area_m2:.2f} m^2  +/- {ro.floor_area_ci_m2:.2f}")
    for i, (L, ci) in enumerate(zip(ro.wall_lengths_m, ro.wall_length_ci_m)):
        print(f"  wall {i}: {L:.2f} m  +/- {ci*100:.1f} cm")
    print(f"Perimeter: {sum(ro.wall_lengths_m):.2f} m")
