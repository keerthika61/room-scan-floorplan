"""
planes.py
---------
Find the floor and (if present) the ceiling, and compute ceiling height.

Because the data is gravity-aligned (Y is up), the floor and ceiling are
horizontal slabs: large groups of points sharing nearly the same Y value.
So instead of a heavy plane-fitting routine, we build a fine histogram of
the Y coordinate and look for the dominant peaks:

  - The FLOOR is the strong low peak (most points: people scan floors a lot).
  - The CEILING, when the room was scanned looking up, is a strong high peak
    well above the floor.

Working in 1-D along the known vertical axis is fast, robust to furniture
clutter (chairs add points but don't form a floor-sized horizontal slab),
and gives a natural way to express uncertainty: the spread of the slab's
points around its peak.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# A horizontal slab is "found" only if at least this fraction of all points
# fall within the slab band. Keeps us from calling a tabletop a ceiling.
MIN_SLAB_FRACTION = 0.03
# Half-thickness (m) of the band we count around a candidate height.
SLAB_HALF_BAND_M = 0.05
# A ceiling must be at least this far above the floor to be believable.
MIN_ROOM_HEIGHT_M = 1.8
MAX_ROOM_HEIGHT_M = 4.5


@dataclass
class Slab:
    """A horizontal surface found at some height."""

    height: float          # Y value of the slab (meters)
    point_count: int       # how many points fell in its band
    spread: float          # std of point Y within the band (meters) -> uncertainty


@dataclass
class HeightResult:
    floor: Slab
    ceiling: Slab | None
    ceiling_height_m: float | None
    # 95%-ish half-width on the height estimate (meters). None if no ceiling.
    ceiling_height_ci_m: float | None
    note: str


def _histogram_peaks(y: np.ndarray, bin_size: float = 0.01):
    """Return (bin_centers, counts) for a fine 1-D histogram of Y."""
    lo, hi = float(y.min()), float(y.max())
    nbins = max(10, int(np.ceil((hi - lo) / bin_size)))
    counts, edges = np.histogram(y, bins=nbins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts


def _slab_at(y: np.ndarray, height: float) -> Slab:
    """Measure the slab (count + spread) in a band around a given height."""
    band = np.abs(y - height) <= SLAB_HALF_BAND_M
    pts = y[band]
    spread = float(pts.std()) if pts.size > 1 else SLAB_HALF_BAND_M
    return Slab(height=float(np.median(pts)) if pts.size else height,
                point_count=int(pts.size), spread=spread)


def find_floor_ceiling(points: np.ndarray) -> HeightResult:
    """Find floor + optional ceiling from a world-space point cloud (N,3).

    Y is the vertical axis. Returns a HeightResult with ceiling height and a
    confidence interval when a ceiling is actually present.
    """
    y = points[:, 1]
    total = y.size
    centers, counts = _histogram_peaks(y)

    # Floor: the strongest peak in the lower half of the height range.
    # Most captures oversample the floor, so the global max is almost always
    # the floor; we restrict to the lower 60% of the range to be safe.
    y_lo, y_hi = y.min(), y.max()
    lower_mask = centers <= y_lo + 0.6 * (y_hi - y_lo)
    floor_center = centers[lower_mask][np.argmax(counts[lower_mask])]
    floor = _slab_at(y, floor_center)

    # Ceiling candidate: strongest peak that sits at a believable room height
    # above the floor.
    ceil_mask = (centers >= floor.height + MIN_ROOM_HEIGHT_M) & (
        centers <= floor.height + MAX_ROOM_HEIGHT_M
    )
    ceiling = None
    ceiling_height = None
    ceiling_ci = None
    note = ""

    if ceil_mask.any():
        cand_counts = counts[ceil_mask]
        cand_centers = centers[ceil_mask]
        best = cand_centers[np.argmax(cand_counts)]
        slab = _slab_at(y, best)
        if slab.point_count >= MIN_SLAB_FRACTION * total:
            ceiling = slab
            ceiling_height = ceiling.height - floor.height
            # Combine the two slabs' spreads in quadrature as the height
            # uncertainty, then widen slightly for discretisation.
            ceiling_ci = float(np.sqrt(floor.spread**2 + ceiling.spread**2) + 0.005)
            note = "ceiling found"
        else:
            note = "ceiling candidate too sparse; reporting floor only"
    else:
        note = "no ceiling captured (floor-only scan)"

    return HeightResult(
        floor=floor,
        ceiling=ceiling,
        ceiling_height_m=ceiling_height,
        ceiling_height_ci_m=ceiling_ci,
        note=note,
    )


if __name__ == "__main__":
    import sys
    import open3d as o3d

    ply = sys.argv[1] if len(sys.argv) > 1 else "outputs/c00a170fe1/cloud.ply"
    pts = np.asarray(o3d.io.read_point_cloud(ply).points)
    res = find_floor_ceiling(pts)
    print(f"Floor:   Y={res.floor.height:.3f} m  ({res.floor.point_count:,} pts, spread {res.floor.spread*100:.1f} cm)")
    if res.ceiling:
        print(f"Ceiling: Y={res.ceiling.height:.3f} m  ({res.ceiling.point_count:,} pts, spread {res.ceiling.spread*100:.1f} cm)")
        print(f"Ceiling height = {res.ceiling_height_m:.3f} m  +/- {res.ceiling_height_ci_m*100:.1f} cm")
    else:
        print("Ceiling: not found")
    print(f"Note: {res.note}")
