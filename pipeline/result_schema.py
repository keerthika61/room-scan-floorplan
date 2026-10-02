"""
result_schema.py
----------------
Assemble the machine-readable result for one capture.

Every measurement carries a value, a unit, and a confidence interval, because
the case study scores calibration: a number without an honest +/- is not an
answer. The schema is intentionally small and stable so a reviewer can diff
two runs easily.
"""

from __future__ import annotations

from typing import Any

SCHEMA_VERSION = "1.0"


def _measure(value, ci, unit, method, note=""):
    """One measurement record: value + confidence interval + provenance."""
    return {
        "value": None if value is None else round(float(value), 4),
        "ci_half_width": None if ci is None else round(float(ci), 4),
        "unit": unit,
        "method": method,
        "note": note,
    }


def build_result(scan_name: str, height, outline, n_points: int) -> dict[str, Any]:
    """Compose the full result dict for a scan from the stage outputs."""
    walls = [
        {
            "index": i,
            "length": _measure(L, ci, "m", "rectilinear_outline"),
        }
        for i, (L, ci) in enumerate(
            zip(outline.wall_lengths_m, outline.wall_length_ci_m)
        )
    ]

    ceiling = _measure(
        height.ceiling_height_m,
        height.ceiling_height_ci_m,
        "m",
        "robust_plane_fit_standard_error",
        note=height.note,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "scan": scan_name,
        "tier": "lidar",
        "point_count": int(n_points),
        "room": {
            "floor_area": _measure(
                outline.floor_area_m2, outline.floor_area_ci_m2, "m^2",
                "segmented_room_polygon",
            ),
            "ceiling_height": ceiling,
            "wall_count": len(walls),
            "perimeter": _measure(
                sum(outline.wall_lengths_m),
                sum(outline.wall_length_ci_m),
                "m",
                "sum_of_wall_lengths",
            ),
            "orientation_deg": round(float(outline.orientation_deg), 2),
            "segmentation_mode": outline.segmentation_mode,
            "walls": walls,
            "outline_polygon_xz": [
                [round(float(x), 4), round(float(z), 4)]
                for x, z in outline.polygon_xz
            ],
        },
        "known_limitations": [
            "Per-room segmentation isolates the dominant enclosed room by "
            "breaking narrow doorway necks; captures with no single compact "
            "room fall back to the full footprint (see segmentation_mode).",
            "Openings (doors/windows) detection not yet implemented at this tier.",
        ],
    }
