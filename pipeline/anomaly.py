"""
anomaly.py
----------
Unsupervised surface-anomaly indicator (a scaffold for damage detection).

Honest scope
============
The brief asks for damage detection (stains, cracks, classes, extents). A
trained classifier needs labeled damage, and the sample captures contain
**no staged damage** — so a trained/validated classifier cannot be built or
tested here. Rather than fake one, this module provides the unsupervised
*indicator* a real damage pipeline would start from, and reports honestly when
nothing anomalous is present.

Method: on an RGB frame, discolouration shows up as pixels whose colour
(chroma in CIE-Lab) deviates strongly from the surface's own distribution. We
flag connected regions of high chroma-deviation above a robust threshold and
report their pixel extent. On clean input this correctly returns few/no
regions.

This is explicitly an INDICATOR, not a damage classifier: no class labels, no
metric real-world extent (that would need the depth of each flagged pixel),
and it is not claimed to be validated.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class AnomalyRegion:
    area_px: int
    bbox: tuple[int, int, int, int]  # x, y, w, h
    mean_chroma_dev: float


# A pixel is "anomalous" if its chroma deviates more than this many robust
# sigmas from the frame's median chroma.
CHROMA_SIGMA = 4.0
MIN_REGION_PX = 400  # ignore specks


def find_anomalies(rgb: np.ndarray) -> list[AnomalyRegion]:
    """Return high-chroma-deviation regions in an RGB frame (indicator only)."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    a = lab[:, :, 1] - 128.0
    b = lab[:, :, 2] - 128.0
    chroma = np.sqrt(a * a + b * b)

    med = np.median(chroma)
    # robust sigma via MAD, floored so a near-uniform frame doesn't produce
    # absurd (divide-by-tiny) deviation scores.
    mad = np.median(np.abs(chroma - med))
    sigma = max(1.4826 * mad, 1.0)
    mask = (chroma - med) > CHROMA_SIGMA * sigma
    mask = mask.astype(np.uint8)

    # clean up and group
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    regions = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < MIN_REGION_PX:
            continue
        x, y, w, h = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                      int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
        dev = float((chroma[labels == i] - med).mean() / sigma)
        regions.append(AnomalyRegion(area_px=area, bbox=(x, y, w, h),
                                     mean_chroma_dev=round(dev, 2)))
    return regions


if __name__ == "__main__":
    import sys
    from .video import read_frame

    scan = sys.argv[1] if len(sys.argv) > 1 else "sample_data/single_room/c00a170fe1"
    idx = int(sys.argv[2]) if len(sys.argv) > 2 else 800
    rgb = read_frame(scan, idx)
    regions = find_anomalies(rgb)
    print(f"frame {idx}: {len(regions)} anomalous region(s) (indicator, not a "
          f"validated damage classifier)")
    for r in regions:
        print(f"  area {r.area_px}px, bbox {r.bbox}, chroma dev {r.mean_chroma_dev} sigma")
    if not regions:
        print("  none above threshold -> consistent with undamaged surfaces")
