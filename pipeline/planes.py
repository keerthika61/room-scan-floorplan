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

# A horizontal slab counts as a ceiling if it holds at least this fraction of
# all points OR this many absolute points. People scan ceilings far less than
# floors, so a partially-captured ceiling can be a small fraction yet still be
# a real, large horizontal surface. Using an absolute floor on the count makes
# detection stable across different frame-sampling densities.
MIN_SLAB_FRACTION = 0.01
MIN_SLAB_POINTS = 8000
# Half-thickness (m) of the band we count around a candidate height.
SLAB_HALF_BAND_M = 0.05
# A ceiling must be at least this far above the floor to be believable.
# Real rooms are rarely under ~2.2 m; a strong horizontal surface at 1.8-2.1 m
# is almost always a shelf, counter, or mezzanine edge, not a ceiling. Using
# 2.2 m avoids calling such surfaces a ceiling (a real false positive we hit
# on the floor-only scan, which reported a bogus 1.81 m "ceiling").
MIN_ROOM_HEIGHT_M = 2.2
MAX_ROOM_HEIGHT_M = 4.5


@dataclass
class Slab:
    """A horizontal surface found at some height."""

    height: float          # Y value of the slab (meters)
    point_count: int       # how many points fell in its band
    spread: float          # std of point Y within the band (meters)
    height_se: float = 0.0  # standard error of the slab height (meters)


# Sensor systematic floor on any single-plane height estimate (meters).
# iPhone LiDAR depth has a small bias that no amount of averaging removes, so
# the CI should never claim to be tighter than this.
PLANE_SYSTEMATIC_M = 0.005


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
    """Fit a horizontal plane to the slab and report height + its std error.

    Previously this returned the raw std of all band points as the
    "uncertainty". That conflates the slab's physical thickness/roughness with
    the uncertainty in WHERE the plane sits, which inflates the CI.

    The quantity we actually want is how well we know the plane's height. For
    the mean of N measurements that is the standard error = std / sqrt(N),
    estimated after a robust trim so a few stray points (light fixtures,
    vents, people) don't drag the plane. We still keep `spread` for context.
    """
    band = np.abs(y - height) <= SLAB_HALF_BAND_M
    pts = y[band]
    if pts.size < 2:
        return Slab(height=height, point_count=int(pts.size),
                    spread=SLAB_HALF_BAND_M, height_se=SLAB_HALF_BAND_M)

    # Iterative robust trim: re-center on the median, keep points within
    # 2.5 sigma, repeat a couple of times. This fits the dominant plane and
    # discards outliers that would otherwise bias both height and spread.
    keep = pts
    center = float(np.median(keep))
    for _ in range(3):
        s = keep.std()
        if s == 0:
            break
        m = np.abs(keep - center) <= 2.5 * s
        if m.sum() < 10 or m.all():
            keep = keep[m]
            center = float(keep.mean())
            break
        keep = keep[m]
        center = float(keep.mean())

    spread = float(keep.std()) if keep.size > 1 else 0.0
    # Standard error of the plane height, floored by the sensor systematic so
    # we never over-claim precision just because N is huge.
    se = spread / np.sqrt(max(keep.size, 1))
    se = float(np.sqrt(se**2 + PLANE_SYSTEMATIC_M**2))
    return Slab(height=center, point_count=int(keep.size),
                spread=spread, height_se=se)


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
        if slab.point_count >= min(MIN_SLAB_FRACTION * total, MIN_SLAB_POINTS):
            ceiling = slab
            ceiling_height = ceiling.height - floor.height
            # Ceiling height is a difference of two independent plane heights,
            # so its uncertainty is the quadrature sum of their standard
            # errors. Reported as a ~2-sigma (95%) half-width.
            combined_se = np.sqrt(floor.height_se**2 + ceiling.height_se**2)
            ceiling_ci = float(2.0 * combined_se)
            note = "ceiling found (least-squares plane fit)"
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
