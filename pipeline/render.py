"""
render.py
---------
Render a top-down floor plan image from the recovered geometry.

Draws:
  - the wall-height point density (faint, for context),
  - the recovered room outline polygon (walls),
  - wall length labels,
  - a header with ceiling height and floor area + their confidence intervals.

The image is a human-facing artifact; the JSON (see result_schema.py) is the
machine-facing one. Both come from the same numbers.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render_floor_plan(
    points: np.ndarray,
    floor_y: float,
    outline,                 # RoomOutline
    height,                  # HeightResult
    out_path: str | Path,
    title: str = "",
    openings=None,           # list[Opening] or None
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    poly = np.vstack([outline.polygon_xz, outline.polygon_xz[:1]])

    fig, ax = plt.subplots(figsize=(9, 9))

    # Faint wall-height context.
    band = (points[:, 1] >= floor_y + 0.3) & (points[:, 1] <= floor_y + 1.5)
    w = points[band]
    if w.shape[0]:
        ax.scatter(w[:, 0], w[:, 2], s=0.15, c="#d9d9d9", zorder=1)

    # Outline (walls).
    ax.plot(poly[:, 0], poly[:, 1], "-", color="#c0392b", lw=2.0, zorder=3)
    ax.scatter(outline.polygon_xz[:, 0], outline.polygon_xz[:, 1],
               c="#2c3e50", s=10, zorder=4)

    # Wall length labels on edges longer than 0.5 m (keeps it readable).
    for i, L in enumerate(outline.wall_lengths_m):
        if L < 0.5:
            continue
        a = outline.polygon_xz[i]
        b = outline.polygon_xz[(i + 1) % len(outline.polygon_xz)]
        mid = 0.5 * (a + b)
        ax.text(mid[0], mid[1], f"{L:.2f} m", fontsize=7,
                color="#2c3e50", ha="center", va="center", zorder=5,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))

    # Openings (doors/windows): mark their centers and widths.
    for o in (openings or []):
        cx, cz = o.center_xz
        color = "#2980b9" if o.kind == "door" else "#16a085"
        ax.scatter([cx], [cz], c=color, s=60, marker="s", zorder=6,
                   edgecolors="white", linewidths=0.8)
        ax.text(cx, cz, f"  {o.kind} {o.width_m:.2f}m", fontsize=7,
                color=color, ha="left", va="center", zorder=6)

    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")

    ch = height.ceiling_height_m
    ch_txt = (
        f"ceiling height {ch:.2f} m +/- {height.ceiling_height_ci_m*100:.1f} cm"
        if ch is not None
        else "ceiling: not captured (floor-only scan)"
    )
    header = (
        f"{title}\n"
        f"floor area {outline.floor_area_m2:.2f} m^2 +/- {outline.floor_area_ci_m2:.2f}   |   "
        f"{ch_txt}   |   outline rotation {outline.orientation_deg:.0f} deg"
    )
    ax.set_title(header, fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
