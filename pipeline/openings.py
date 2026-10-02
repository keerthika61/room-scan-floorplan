"""
openings.py
-----------
Detect openings (doors / wide windows) along the room outline and measure
their widths.

Idea
====
A wall is a line of dense points at wall height. A doorway or window is a
GAP in that line: as you walk along a wall edge, the point density stays high
over solid wall and drops to near zero across an opening. So for each wall
edge of the outline we:

  1. sample points in a thin strip hugging that edge (at wall height),
  2. project them onto the edge to get a 1-D coverage profile,
  3. mark stretches with little/no coverage as candidate openings,
  4. keep stretches whose width is door/window-sized and report them.

Width uncertainty is tied to the sampling bin, consistent with the rest of
the pipeline's calibrated-interval approach.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WALL_BAND_LOW_M = 0.1       # strip spans this far above floor ...
WALL_BAND_HIGH_M = 2.1      # ... to here (covers doors and most windows)
STRIP_HALF_WIDTH_M = 0.15   # how far either side of the edge line to look
BIN_M = 0.05                # along-edge sampling resolution
MIN_COVER_PTS = 3           # a bin with >= this many points counts as "wall"
MIN_OPENING_M = 0.6         # narrower than a door -> ignore (clutter gap)
MAX_OPENING_M = 2.5         # wider -> probably not a single opening / is a corner


@dataclass
class Opening:
    wall_index: int
    width_m: float
    width_ci_m: float
    center_xz: tuple[float, float]
    kind: str  # "door" or "window-or-wide" (heuristic by width)


def _edge_openings(a, b, pts_xz, wall_index) -> list[Opening]:
    """Find openings along one wall edge a->b."""
    edge = b - a
    length = float(np.linalg.norm(edge))
    if length < MIN_OPENING_M:
        return []
    t_hat = edge / length                 # unit vector along the edge
    n_hat = np.array([-t_hat[1], t_hat[0]])  # unit normal to the edge

    # Coordinates of every point relative to the edge: s = along, d = across.
    rel = pts_xz - a
    s = rel @ t_hat
    d = rel @ n_hat
    near = (np.abs(d) <= STRIP_HALF_WIDTH_M) & (s >= 0) & (s <= length)
    s = s[near]

    nbins = max(1, int(np.ceil(length / BIN_M)))
    counts, edges = np.histogram(s, bins=nbins, range=(0, length))
    is_wall = counts >= MIN_COVER_PTS      # True where there is solid wall

    # Find contiguous runs of "not wall" (gaps) that are bounded by wall on
    # BOTH sides (so we don't count the open ends of a wall as openings).
    openings: list[Opening] = []
    i = 0
    while i < nbins:
        if not is_wall[i]:
            j = i
            while j < nbins and not is_wall[j]:
                j += 1
            # gap spans bins [i, j); require wall before i and at/after j
            bounded = (i > 0 and is_wall[i - 1]) and (j < nbins and is_wall[j])
            width = (j - i) * BIN_M
            if bounded and MIN_OPENING_M <= width <= MAX_OPENING_M:
                mid_s = (i + j) * 0.5 * BIN_M
                center = a + t_hat * mid_s
                kind = "door" if width <= 1.2 else "window-or-wide"
                openings.append(Opening(
                    wall_index=wall_index,
                    width_m=float(width),
                    width_ci_m=float(BIN_M + 0.01 * width),
                    center_xz=(float(center[0]), float(center[1])),
                    kind=kind,
                ))
            i = j
        else:
            i += 1
    return openings


def detect_openings(points: np.ndarray, floor_y: float, outline) -> list[Opening]:
    """Detect openings along every wall of the outline polygon."""
    band = (points[:, 1] >= floor_y + WALL_BAND_LOW_M) & (
        points[:, 1] <= floor_y + WALL_BAND_HIGH_M
    )
    pts_xz = points[band][:, [0, 2]]
    if pts_xz.shape[0] < 50:
        return []

    poly = outline.polygon_xz
    k = len(poly)
    found: list[Opening] = []
    for i in range(k):
        a = poly[i]
        b = poly[(i + 1) % k]
        found.extend(_edge_openings(a, b, pts_xz, i))
    return found


if __name__ == "__main__":
    import sys
    import open3d as o3d
    from .planes import find_floor_ceiling
    from .room_outline import compute_room_outline

    ply = sys.argv[1] if len(sys.argv) > 1 else "outputs/c7d28f72c6/cloud.ply"
    pts = np.asarray(o3d.io.read_point_cloud(ply).points)
    floor = find_floor_ceiling(pts).floor.height
    outline = compute_room_outline(pts, floor)
    ops = detect_openings(pts, floor, outline)
    print(f"Detected {len(ops)} openings:")
    for o in ops:
        print(f"  wall {o.wall_index}: {o.kind} width {o.width_m:.2f} m "
              f"+/- {o.width_ci_m*100:.0f} cm at ({o.center_xz[0]:.1f}, {o.center_xz[1]:.1f})")
