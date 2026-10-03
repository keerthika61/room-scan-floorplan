"""
room_segment.py
---------------
Extract the DOMINANT enclosed room from a floor cloud that may also contain
regions seen through open doorways.

Problem
=======
`room_outline.py` traces the full floor-coverage blob. When a scan sees
through an open door into a hallway or neighbouring room, that blob grows a
thin "tendril" into the next space and the outline wraps it too, so the
reported room is too big and the wrong shape.

Idea
====
The real room is a large, compact, wall-enclosed area. Through-doorway leaks
connect to it only through narrow necks (a doorway is ~0.8-1.0 m wide). If we
erode the filled floor region by a little more than half a doorway width, the
narrow necks snap and the leaked regions fall off as separate blobs; the
largest surviving blob is the room core. Dilating back restores its true size.

This is a morphological opening with a doorway-sized kernel, then keep the
largest connected component. Simple, fast, and makes no rectangular-room
assumption.
"""

from __future__ import annotations

import cv2
import numpy as np

GRID_RES_M = 0.02
# A doorway is ~0.8-1.0 m. Eroding by a bit over half that (~0.55 m) snaps
# doorway necks while keeping genuinely wide room openings connected.
DOORWAY_BREAK_M = 0.55


def _fill_floor_mask(xz: np.ndarray, res_m: float):
    """Rasterize floor points, close gaps, fill holes -> solid room mask."""
    mn = xz.min(0) - 0.15
    size = np.ceil((xz.max(0) + 0.15 - mn) / res_m).astype(int) + 1
    grid = np.zeros((size[1], size[0]), np.uint8)
    cols = np.clip(((xz[:, 0] - mn[0]) / res_m).astype(int), 0, size[0] - 1)
    rows = np.clip(((xz[:, 1] - mn[1]) / res_m).astype(int), 0, size[1] - 1)
    grid[rows, cols] = 255

    # Close small gaps in floor coverage so the room is one solid blob.
    k = np.ones((9, 9), np.uint8)
    closed = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, k)

    # Fill interior holes (furniture footprints) via background flood fill.
    h, w = closed.shape
    ff = closed.copy()
    ffmask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, ffmask, (0, 0), 255)
    filled = closed | cv2.bitwise_not(ff)
    return filled, mn


def _dominant_component(mask: np.ndarray) -> np.ndarray:
    """Open with a doorway-sized kernel, keep the largest blob, dilate back."""
    r = max(1, int(round(DOORWAY_BREAK_M / GRID_RES_M)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))

    eroded = cv2.erode(mask, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(eroded, connectivity=8)
    if n <= 1:
        # Erosion removed everything (tiny/odd room) -> fall back to input.
        return mask
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    core = np.where(labels == biggest, 255, 0).astype(np.uint8)

    # Grow the core back to its original extent, but clip to the original
    # mask so we never invent floor the scan did not see.
    regrown = cv2.dilate(core, kernel)
    return (regrown & mask).astype(np.uint8)


# If the dominant room is this small a share of the full coverage, the scan
# probably has no single compact room (e.g. a corridor-heavy capture). In that
# case segmentation would throw away most of the real space, so we fall back
# to the full footprint and report that honestly.
MIN_DOMINANT_FRACTION = 0.35


def dominant_room_mask(points: np.ndarray, floor_y: float,
                       footprint_band_m: float = 0.12):
    """Return (room_mask, origin_xz, res_m, info) for the dominant room.

    `points` are world-frame. The method erodes the filled floor by a
    doorway-sized kernel to snap narrow necks, keeps the largest blob, and
    dilates it back. If that dominant blob is too small a share of the whole
    (no clear compact room), it falls back to the full footprint and says so
    in `info["mode"]`.
    """
    band = (points[:, 1] >= floor_y - 0.05) & (points[:, 1] <= floor_y + footprint_band_m)
    xz = points[band][:, [0, 2]]
    if xz.shape[0] < 100:
        xz = points[:, [0, 2]]

    filled, origin = _fill_floor_mask(xz, GRID_RES_M)
    room = _dominant_component(filled)

    full_cells = int((filled > 0).sum())
    room_cells = int((room > 0).sum())
    frac = room_cells / max(full_cells, 1)

    if frac < MIN_DOMINANT_FRACTION:
        info = {"mode": "full_footprint_fallback", "dominant_fraction": round(frac, 3)}
        return filled, origin, GRID_RES_M, info

    info = {"mode": "dominant_room", "dominant_fraction": round(frac, 3)}
    return room, origin, GRID_RES_M, info


# A separated component counts as a distinct room only if it is at least this
# big after the doorway-break erosion. Smaller fragments are noise or alcoves.
MIN_ROOM_AREA_M2 = 2.0


def count_separable_rooms(points: np.ndarray, floor_y: float,
                          footprint_band_m: float = 0.12) -> dict:
    """How many distinct rooms does this capture actually contain?

    After eroding the filled floor by a doorway-sized kernel, genuinely
    separate rooms survive as separate blobs. Counting the blobs that are
    room-sized tells us honestly whether a capture is single-room (one blob)
    or multi-room (several). The pipeline reports one room when this is 1; a
    true multi-room capture (e.g. at the walk-in test) would report >1 here,
    which is the hook a future multi-room stitch would build on.
    """
    band = (points[:, 1] >= floor_y - 0.05) & (points[:, 1] <= floor_y + footprint_band_m)
    xz = points[band][:, [0, 2]]
    if xz.shape[0] < 100:
        xz = points[:, [0, 2]]

    filled, _ = _fill_floor_mask(xz, GRID_RES_M)
    r = max(1, int(round(DOORWAY_BREAK_M / GRID_RES_M)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    eroded = cv2.erode(filled, kernel)

    n, _, stats, _ = cv2.connectedComponentsWithStats(eroded, connectivity=8)
    cell_area = GRID_RES_M * GRID_RES_M
    room_areas = sorted(
        (float(stats[i, cv2.CC_STAT_AREA]) * cell_area for i in range(1, n)),
        reverse=True,
    )
    rooms = [a for a in room_areas if a >= MIN_ROOM_AREA_M2]
    return {
        "room_count": len(rooms),
        "room_core_areas_m2": [round(a, 2) for a in rooms],
        "is_multi_room": len(rooms) > 1,
    }


if __name__ == "__main__":
    import sys
    import open3d as o3d
    from .planes import find_floor_ceiling

    ply = sys.argv[1] if len(sys.argv) > 1 else "outputs/c00a170fe1/cloud.ply"
    pts = np.asarray(o3d.io.read_point_cloud(ply).points)
    floor = find_floor_ceiling(pts).floor.height

    full, _ = _fill_floor_mask(pts[(pts[:, 1] >= floor - 0.05) &
                                   (pts[:, 1] <= floor + 0.12)][:, [0, 2]], GRID_RES_M)
    room, origin, res, info = dominant_room_mask(pts, floor)

    full_area = int((full > 0).sum()) * res * res
    room_area = int((room > 0).sum()) * res * res
    print(f"full floor-coverage area:  {full_area:.2f} m^2")
    print(f"reported room area:        {room_area:.2f} m^2")
    print(f"mode: {info['mode']}  (dominant fraction {info['dominant_fraction']})")
