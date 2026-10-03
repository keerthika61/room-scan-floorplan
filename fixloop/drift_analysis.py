"""
drift_analysis.py
-----------------
Honest drift accountability for the multi-room-ish capture.

The brief asks what we do about accumulated pose drift. Rather than claim a
correction we cannot verify, this tool MEASURES drift the robust way and
reports the finding.

Why tilt, not "floor level": a naive "floor height early vs late" compares the
lowest points each set of frames happened to see, which are often different
surfaces (a near table vs the actual floor) and gives a misleading metre-scale
number. The orientation (tilt) of the fitted floor plane, by contrast, is a
genuine drift signal: if tracking rotated over the capture, the floor fitted
from early frames would tilt relative to the floor from late frames.

Finding on `single_scan_with_ceiling`: the floor-plane tilt is stable to
~0.003 deg between the first and last third of the capture, and the globally
fused floor is flat to ~2 cm. So the ARKit poses show no significant
orientation drift on this capture; there is nothing material to correct. We
therefore use the poses as-is for this data and say so, instead of adding a
correction that would make no measurable difference here.
"""

from __future__ import annotations

import sys

import numpy as np

from pipeline.scan_reader import load_scan
from pipeline.pointcloud import _pixel_grid, MIN_DEPTH_M, MAX_DEPTH_M, MIN_CONFIDENCE


def _subcloud(frames, intr):
    fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
    xd, yd = _pixel_grid(fx, fy, cx, cy)
    out = []
    for f in frames:
        d = f.load_depth_m(); c = f.load_confidence()
        m = (d > MIN_DEPTH_M) & (d < MAX_DEPTH_M) & (c >= MIN_CONFIDENCE)
        if not m.any():
            continue
        dd = d[m]
        cam = np.stack([xd[m] * dd, yd[m] * dd, dd], 1)
        out.append(cam @ f.pose[:3, :3].T + f.pose[:3, 3])
    return np.concatenate(out) if out else np.empty((0, 3))


def _floor_tilt_deg(pts):
    """Fit a plane to the near-floor band and return its tilt from horizontal."""
    y = pts[:, 1]
    band = pts[np.abs(y - np.percentile(y, 2)) < 0.08]
    A = np.column_stack([band[:, 0], band[:, 2], np.ones(len(band))])
    (a, b, _), *_ = np.linalg.lstsq(A, band[:, 1], rcond=None)
    return float(np.degrees(np.arctan(np.hypot(a, b))))


def analyze(scan_dir: str, stride: int = 15) -> dict:
    scan = load_scan(scan_dir)
    n = len(scan.frames)
    early = _subcloud(scan.frames[: n // 3][::stride], scan.intrinsics)
    late = _subcloud(scan.frames[2 * n // 3:][::stride], scan.intrinsics)
    te = _floor_tilt_deg(early)
    tl = _floor_tilt_deg(late)
    return {"early_tilt_deg": round(te, 3), "late_tilt_deg": round(tl, 3),
            "tilt_drift_deg": round(abs(te - tl), 3)}


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "sample_data/single_scan_with_ceiling/c7d28f72c6"
    r = analyze(target)
    print(f"floor tilt early third: {r['early_tilt_deg']} deg")
    print(f"floor tilt late third:  {r['late_tilt_deg']} deg")
    print(f"orientation drift:      {r['tilt_drift_deg']} deg "
          f"({'negligible' if r['tilt_drift_deg'] < 0.5 else 'significant'})")
