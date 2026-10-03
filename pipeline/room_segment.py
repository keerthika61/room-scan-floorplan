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
largest surviving blob identifies the room core.

Erosion is used only to SELECT the right room. The room's extent is then taken
from the matching connected component of the UN-eroded mask -- dilating the
eroded core back would under-size the room by ~the kernel radius at every
boundary (a bug an earlier version had; see git history / `TestKnownRoomAccuracy`).
Simple, fast, and makes no rectangular-room assumption.
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
    """Erode to break doorway necks, pick the largest blob, then recover that
    room's full extent from the original (un-eroded) mask."""
    r = max(1, int(round(DOORWAY_BREAK_M / GRID_RES_M)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))

    eroded = cv2.erode(mask, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(eroded, connectivity=8)
    if n <= 1:
        # Erosion removed everything (tiny/odd room) -> fall back to input.
        return mask
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    core = (labels == biggest).astype(np.uint8)

    # Erosion is used only to SELECT the right room (break doorway necks and
    # pick the dominant blob). To avoid shrinking the room, recover its exact
    # ORIGINAL extent: keep whichever connected component of the untouched
    # `mask` the core overlaps. Dilating the eroded core back would under-size
    # the room by ~the kernel radius at every boundary; this does not.
    n2, labels2, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    overlap_labels = labels2[core.astype(bool)]
    overlap_labels = overlap_labels[overlap_labels > 0]
    if overlap_labels.size == 0:
        return mask
    room_label = np.bincount(overlap_labels).argmax()
    return np.where(labels2 == room_label, 255, 0).astype(np.uint8)


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
    recovers that room's full extent from the un-eroded mask. If the dominant
    room is too small a share of the whole (no clear compact room), it falls
    back to the full footprint and says so in `info["mode"]`.
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


def label_all_rooms(points: np.ndarray, floor_y: float,
                    footprint_band_m: float = 0.12):
    """Return a per-room mask for EVERY room-sized region, plus the shared grid.

    Generalises `dominant_room_mask` to multi-room captures: erode to break
    doorway necks, then for each surviving room-sized core recover its full
    extent from the un-eroded mask (same no-shrink trick as the single-room
    path). Returns (list_of_room_masks, origin_xz, res_m). A single-room
    capture yields one mask; a true multi-room capture yields several, which
    is what a whole-property stitch would consume.
    """
    band = (points[:, 1] >= floor_y - 0.05) & (points[:, 1] <= floor_y + footprint_band_m)
    xz = points[band][:, [0, 2]]
    if xz.shape[0] < 100:
        xz = points[:, [0, 2]]

    filled, origin = _fill_floor_mask(xz, GRID_RES_M)
    r = max(1, int(round(DOORWAY_BREAK_M / GRID_RES_M)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    eroded = cv2.erode(filled, kernel)

    n, core_labels, stats, _ = cv2.connectedComponentsWithStats(eroded, connectivity=8)
    cell_area = GRID_RES_M * GRID_RES_M

    # Room-sized cores, largest first.
    cores = [i for i in sorted(range(1, n), key=lambda i: -stats[i, cv2.CC_STAT_AREA])
             if stats[i, cv2.CC_STAT_AREA] * cell_area >= MIN_ROOM_AREA_M2]
    if not cores:
        return [filled], origin, GRID_RES_M
    if len(cores) == 1:
        # Single room: recover the whole connected component it sits in.
        return [_dominant_component(filled)], origin, GRID_RES_M

    # Multiple rooms that may be MERGED in the filled mask (doorway bridged by
    # the morphological close). Split every filled pixel to its nearest core
    # via a distance transform from each core, so a shared blob is divided
    # along the doorway rather than collapsed into one room.
    h, w = filled.shape
    best_dist = np.full((h, w), np.inf, np.float32)
    assign = np.zeros((h, w), np.int32)
    for ridx, i in enumerate(cores, start=1):
        seed = np.ones((h, w), np.uint8)
        seed[core_labels == i] = 0
        dist = cv2.distanceTransform(seed, cv2.DIST_L2, 3)
        closer = dist < best_dist
        best_dist[closer] = dist[closer]
        assign[closer] = ridx

    masks = []
    for ridx in range(1, len(cores) + 1):
        m = ((filled > 0) & (assign == ridx)).astype(np.uint8) * 255
        if int((m > 0).sum()) * cell_area >= MIN_ROOM_AREA_M2:
            masks.append(m)
    return (masks or [filled]), origin, GRID_RES_M


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
