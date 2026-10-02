"""
footprint.py
------------
Recover the room footprint (outline polygon), wall lengths and floor area
from a world-space point cloud.

Idea
====
The room's floor is a flat horizontal patch. Looking straight down (the X-Z
plane, since Y is up), the floor points fill the room's 2-D shape. So:

  1. Keep points in a horizontal band just above the floor (the floor slab
     plus a little, which captures the floor extent without furniture tops).
  2. Rasterize those points into a 2-D occupancy grid (a top-down image).
  3. Close small holes, keep the largest connected region -> the room area.
  4. Trace its outer contour and simplify it to a polygon.

The polygon edges are the walls; its area is the floor area. This is robust
to clutter and works for non-rectangular (L-shaped) rooms, which fitting
four wall planes is not.

All distances are in meters. The grid resolution sets the measurement
granularity, so we also use it to express a sensible uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Size of one grid cell (meters). 2 cm balances wall sharpness vs. hole-free
# floor coverage.
GRID_RES_M = 0.02
# Keep points from floor level up to this height when forming the footprint.
# Low enough to exclude most furniture tops, high enough to catch the whole
# floor even with sensor noise.
FOOTPRINT_BAND_M = 0.15
# Morphological closing kernel (cells) to bridge small gaps in floor coverage.
CLOSE_KERNEL_CELLS = 7
# Polygon simplification tolerance (meters). Points within this of a straight
# edge get merged, turning a jagged contour into clean wall segments.
SIMPLIFY_TOL_M = 0.08


@dataclass
class Footprint:
    # Polygon vertices in world X-Z, shape (K, 2), ordered around the room.
    polygon_xz: np.ndarray
    wall_lengths_m: list[float]
    wall_length_ci_m: list[float]
    floor_area_m2: float
    floor_area_ci_m2: float
    # The occupancy grid and its world transform, kept for rendering/openings.
    grid: np.ndarray            # uint8 filled room mask
    grid_origin_xz: np.ndarray  # world (x, z) of grid cell (0, 0)
    grid_res_m: float


def _rasterize_floor(points: np.ndarray, floor_y: float):
    """Build a filled top-down occupancy mask of the floor region."""
    band = (points[:, 1] >= floor_y - 0.05) & (points[:, 1] <= floor_y + FOOTPRINT_BAND_M)
    xz = points[band][:, [0, 2]]
    if xz.shape[0] < 100:
        # Fall back to all points if the band is too thin.
        xz = points[:, [0, 2]]

    mn = xz.min(axis=0) - 0.1
    mx = xz.max(axis=0) + 0.1
    size = np.ceil((mx - mn) / GRID_RES_M).astype(int) + 1
    grid = np.zeros((size[1], size[0]), dtype=np.uint8)  # rows=Z, cols=X

    cols = ((xz[:, 0] - mn[0]) / GRID_RES_M).astype(int)
    rows = ((xz[:, 1] - mn[1]) / GRID_RES_M).astype(int)
    grid[rows, cols] = 255
    return grid, mn


def _largest_filled_region(grid: np.ndarray) -> np.ndarray:
    """Close gaps, fill holes, and keep only the largest blob (the room)."""
    k = np.ones((CLOSE_KERNEL_CELLS, CLOSE_KERNEL_CELLS), np.uint8)
    closed = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, k)

    # Fill interior holes (furniture gaps) by flood-filling the background
    # from a corner and inverting.
    h, w = closed.shape
    ff = closed.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, mask, (0, 0), 255)
    filled = closed | cv2.bitwise_not(ff)

    # Keep the largest connected component.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
    if n <= 1:
        return filled
    biggest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return np.where(labels == biggest, 255, 0).astype(np.uint8)


def _contour_polygon(mask: np.ndarray, origin_xz: np.ndarray):
    """Trace the outer contour of the room mask and simplify to a polygon."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No room contour found")
    cnt = max(contours, key=cv2.contourArea)

    eps = SIMPLIFY_TOL_M / GRID_RES_M
    approx = cv2.approxPolyDP(cnt, eps, closed=True).reshape(-1, 2)

    # Grid cell -> world (x, z).
    poly = np.empty_like(approx, dtype=np.float64)
    poly[:, 0] = origin_xz[0] + approx[:, 0] * GRID_RES_M  # X from column
    poly[:, 1] = origin_xz[1] + approx[:, 1] * GRID_RES_M  # Z from row
    return poly


def _polygon_area(poly: np.ndarray) -> float:
    """Shoelace area of a 2-D polygon."""
    x, z = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(z, -1)) - np.dot(z, np.roll(x, -1)))


def compute_footprint(points: np.ndarray, floor_y: float) -> Footprint:
    """Full footprint recovery from a point cloud and known floor height."""
    grid, origin = _rasterize_floor(points, floor_y)
    room = _largest_filled_region(grid)
    poly = _contour_polygon(room, origin)

    # Wall lengths = polygon edge lengths.
    edges = np.roll(poly, -1, axis=0) - poly
    lengths = np.sqrt((edges**2).sum(axis=1)).tolist()

    # Uncertainty: each endpoint is localized to ~1 grid cell, so a length is
    # uncertain by ~sqrt(2) cells, plus a small fraction for discretization.
    cell = GRID_RES_M
    wall_ci = [float(np.sqrt(2) * cell + 0.01 * L) for L in lengths]

    area = _polygon_area(poly)
    # Area uncertainty: perimeter * cell is a standard first-order estimate of
    # how much a one-cell boundary wobble changes the area.
    perimeter = sum(lengths)
    area_ci = float(perimeter * cell)

    return Footprint(
        polygon_xz=poly,
        wall_lengths_m=lengths,
        wall_length_ci_m=wall_ci,
        floor_area_m2=area,
        floor_area_ci_m2=area_ci,
        grid=room,
        grid_origin_xz=origin,
        grid_res_m=GRID_RES_M,
    )


if __name__ == "__main__":
    import sys
    import open3d as o3d
    from .planes import find_floor_ceiling

    ply = sys.argv[1] if len(sys.argv) > 1 else "outputs/c00a170fe1/cloud.ply"
    pts = np.asarray(o3d.io.read_point_cloud(ply).points)
    floor = find_floor_ceiling(pts).floor.height
    fp = compute_footprint(pts, floor)

    print(f"Room outline: {len(fp.polygon_xz)} vertices")
    print(f"Floor area: {fp.floor_area_m2:.2f} m^2  +/- {fp.floor_area_ci_m2:.2f}")
    print(f"Walls ({len(fp.wall_lengths_m)}):")
    for i, (L, ci) in enumerate(zip(fp.wall_lengths_m, fp.wall_length_ci_m)):
        print(f"  wall {i}: {L:.2f} m  +/- {ci*100:.1f} cm")
    print(f"Perimeter: {sum(fp.wall_lengths_m):.2f} m")
