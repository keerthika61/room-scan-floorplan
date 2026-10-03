"""
measure_ceiling.py
------------------
Standalone measurement used by the fix-loop record. Prints the ceiling-height
estimate and its confidence interval for a given cloud, using the CURRENT
pipeline (whatever planes.find_floor_ceiling does right now).

Run before and after the fix to produce the regenerable before/after numbers.
"""

import sys
import numpy as np
import open3d as o3d

from pipeline.planes import find_floor_ceiling


def main(ply: str) -> None:
    pts = np.asarray(o3d.io.read_point_cloud(ply).points)
    res = find_floor_ceiling(pts)
    print(f"cloud: {ply}")
    print(f"floor_y: {res.floor.height:.4f} m (spread {res.floor.spread*100:.2f} cm)")
    if res.ceiling is not None:
        print(f"ceiling_y: {res.ceiling.height:.4f} m (spread {res.ceiling.spread*100:.2f} cm)")
        print(f"ceiling_height: {res.ceiling_height_m:.4f} m")
        print(f"ceiling_height_ci_half_width: {res.ceiling_height_ci_m*100:.2f} cm")
        gate = 1.5
        status = "PASS" if res.ceiling_height_ci_m * 100 <= gate else "FAIL"
        print(f"gate (<= {gate} cm): {status}")
    else:
        print("ceiling: not found")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "outputs/c7d28f72c6/cloud.ply"
    main(target)
