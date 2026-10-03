"""
benchmark_summary.py
--------------------
Read every outputs/<scan>/result.json and print a compact benchmark table
(and write it to docs/BENCHMARK.md). Lets a reviewer see all reported numbers,
with confidence intervals, at a glance.
"""

import glob
import json
from pathlib import Path


def _fmt(m):
    if m is None or m.get("value") is None:
        return "n/a"
    ci = m.get("ci_half_width")
    ci_txt = f" +/- {ci:.3f}" if ci is not None else ""
    return f"{m['value']:.3f}{ci_txt} {m['unit']}"


def main():
    rows = []
    for f in sorted(glob.glob("outputs/*/result.json")):
        r = json.load(open(f))
        room = r["room"]
        rows.append({
            "scan": r["scan"],
            "points": r["point_count"],
            "mode": room["segmentation_mode"],
            "area": _fmt(room["floor_area"]),
            "ceiling": _fmt(room["ceiling_height"]),
            "walls": room.get("outline_edge_count", room.get("wall_count", "?")),
            "openings": room.get("opening_count", 0),
            "time_s": r.get("timing_seconds", "?"),
        })

    lines = ["# Benchmark Summary", "",
             "Auto-generated from `outputs/*/result.json` by "
             "`python -m fixloop.benchmark_summary`.", "",
             "| Scan | Points | Mode | Floor area | Ceiling height | Outline edges | Openings | Time |",
             "|------|-------:|------|-----------|----------------|------:|---------:|-----:|"]
    for r in rows:
        lines.append(
            f"| {r['scan']} | {r['points']:,} | {r['mode']} | {r['area']} | "
            f"{r['ceiling']} | {r['walls']} | {r['openings']} | {r['time_s']}s |"
        )
    text = "\n".join(lines) + "\n"

    Path("docs").mkdir(exist_ok=True)
    Path("docs/BENCHMARK.md").write_text(text, encoding="utf-8")
    print(text)
    print("Wrote docs/BENCHMARK.md")


if __name__ == "__main__":
    main()
