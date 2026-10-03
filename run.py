"""
run.py -- single-command entry point for one capture.

Usage:
    python run.py <scan_folder> [--stride N] [--out DIR]

Example:
    python run.py sample_data/single_room/c00a170fe1

Produces, under outputs/<scan_name>/:
    cloud.ply        fused 3D point cloud
    result.json      all measurements + confidence intervals (to schema)
    floor_plan.png   rendered top-down plan

One command per capture, as required by the brief.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from pipeline.scan_reader import load_scan
from pipeline.pointcloud import build_point_cloud, save_point_cloud
from pipeline.colorize import build_colored_point_cloud
from pipeline.planes import find_floor_ceiling
from pipeline.room_outline import compute_room_outline
from pipeline.room_segment import count_separable_rooms
from pipeline.openings import detect_openings
from pipeline.render import render_floor_plan
from pipeline.result_schema import build_result


def process(scan_dir: str, stride: int, out_root: str, colorize: bool = False) -> dict:
    t0 = time.time()
    scan = load_scan(scan_dir)
    out_dir = Path(out_root) / scan.name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loaded '{scan.name}' ({len(scan)} frames)")

    pcd = build_point_cloud(scan, frame_stride=stride)
    pts = np.asarray(pcd.points)
    save_point_cloud(pcd, out_dir / "cloud.ply")
    print(f"[2/5] Point cloud: {len(pts):,} points -> cloud.ply")

    height = find_floor_ceiling(pts)
    floor_y = height.floor.height
    print(f"[3/5] Floor/ceiling: {height.note}")

    outline = compute_room_outline(pts, floor_y)
    openings = detect_openings(pts, floor_y, outline)
    rooms = count_separable_rooms(pts, floor_y)
    print(f"[4/5] Outline: {len(outline.polygon_xz)} walls, "
          f"area {outline.floor_area_m2:.2f} m^2, {len(openings)} openings, "
          f"{rooms['room_count']} separable room(s)")

    render_floor_plan(pts, floor_y, outline, height,
                      out_dir / "floor_plan.png", title=scan.name, openings=openings)
    result = build_result(scan.name, height, outline, len(pts),
                          openings=openings, rooms=rooms)
    result["timing_seconds"] = round(time.time() - t0, 1)

    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[5/5] Wrote result.json + floor_plan.png  ({result['timing_seconds']}s)")

    if colorize:
        try:
            cpcd = build_colored_point_cloud(scan, frame_stride=max(stride, 10))
            save_point_cloud(cpcd, out_dir / "cloud_colored.ply")
            print(f"      + cloud_colored.ply ({len(cpcd.points):,} RGB points)")
        except (FileNotFoundError, ValueError) as e:
            print(f"      (colorize skipped: {e})")

    print(f"      -> {out_dir}")
    return result


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Room scan -> floor plan (LiDAR tier)")
    ap.add_argument("scan_folder", help="Path to a scan folder")
    ap.add_argument("--stride", type=int, default=5,
                    help="Use every Nth frame (higher = faster, coarser)")
    ap.add_argument("--out", default="outputs", help="Output root directory")
    ap.add_argument("--colorize", action="store_true",
                    help="Also produce an RGB-colored point cloud from rgb.mp4")
    args = ap.parse_args(argv)

    if not Path(args.scan_folder).is_dir():
        print(f"ERROR: not a folder: {args.scan_folder}", file=sys.stderr)
        return 2

    try:
        process(args.scan_folder, args.stride, args.out, colorize=args.colorize)
    except (FileNotFoundError, ValueError) as e:
        # Expected input problems (missing files, no usable frames): report
        # cleanly instead of dumping a traceback.
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
