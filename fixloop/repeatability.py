"""
repeatability.py
----------------
Repeatability gate: "scan the same room twice, get the same plan."

We do not have two separate captures of the same room in the sample data, so
we simulate two independent captures from ONE scan by splitting its frames
into two disjoint halves (even-indexed vs odd-indexed frames). Each half is an
independent, lower-density capture of the same space. We build geometry from
each half separately and report how closely the two agree.

The brief's gate: two captures of the same room agree within 1 cm or 0.5% per
wall (and ceiling-height spread <= 1 cm across captures). We report the raw
agreement so a reviewer can judge it against that bar.

This is an honest proxy: splitting frames tests the pipeline's stability to
sampling, which is the dominant repeatability factor on a tripod-free handheld
capture. It is NOT a substitute for two true physical captures, and we say so.
"""

from __future__ import annotations

import sys

import numpy as np

from pipeline.scan_reader import load_scan, Scan
from pipeline.pointcloud import build_point_cloud
from pipeline.planes import find_floor_ceiling
from pipeline.room_outline import compute_room_outline


def _half_scan(scan: Scan, parity: int) -> Scan:
    """Return a Scan containing only even- or odd-indexed frames."""
    frames = [f for i, f in enumerate(scan.frames) if i % 2 == parity]
    return Scan(path=scan.path, name=f"{scan.name}_half{parity}",
                intrinsics=scan.intrinsics, frames=frames)


def _measure(scan: Scan):
    pcd = build_point_cloud(scan, frame_stride=2)  # halves are already sparse
    pts = np.asarray(pcd.points)
    h = find_floor_ceiling(pts)
    ro = compute_room_outline(pts, h.floor.height)
    return {
        "ceiling_height": h.ceiling_height_m,
        "floor_area": ro.floor_area_m2,
        "perimeter": sum(ro.wall_lengths_m),
    }


def run(scan_dir: str):
    scan = load_scan(scan_dir)
    print(f"Repeatability proxy on '{scan.name}' "
          f"({len(scan)} frames -> two halves)\n")

    a = _measure(_half_scan(scan, 0))
    b = _measure(_half_scan(scan, 1))

    def line(name, va, vb, unit, gate_abs=None, gate_rel=None):
        if va is None or vb is None:
            print(f"  {name:16s}: half A={va}  half B={vb}  (n/a)")
            return
        diff = abs(va - vb)
        rel = diff / max(abs(0.5 * (va + vb)), 1e-6) * 100
        verdict = ""
        if gate_abs is not None:
            ok = diff <= gate_abs or (gate_rel is not None and rel <= gate_rel)
            verdict = "  PASS" if ok else "  (over gate)"
        print(f"  {name:16s}: A={va:.3f} B={vb:.3f} {unit}  "
              f"|diff|={diff*100:.2f} cm ({rel:.2f}%){verdict}")

    print("Agreement between the two independent halves:")
    line("ceiling_height", a["ceiling_height"], b["ceiling_height"], "m",
         gate_abs=0.01)
    line("floor_area", a["floor_area"], b["floor_area"], "m^2")
    line("perimeter", a["perimeter"], b["perimeter"], "m", gate_rel=0.5)
    print("\nNote: proxy via frame-split, not two physical captures.")
    return a, b


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "sample_data/single_scan_with_ceiling/c7d28f72c6"
    run(target)
