# Compliance Matrix

Honest status of each requirement. Legend: ✅ done · ◻ partial · ❌ not in this build.

| # | Requirement | File / path | Artifact | Status |
|---|-------------|-------------|----------|--------|
| 1 | Capture route (Route 2: stock app + protocol) | `docs/CAPTURE_PROTOCOL.md` | One-page protocol | ✅ |
| 2 | Device matrix (tier → hardware → accuracy) | `docs/CAPTURE_PROTOCOL.md` | Table | ✅ |
| 3 | LiDAR tier pipeline | `pipeline/`, `run.py` | Runs end-to-end | ✅ |
| 4 | Video tier | — | — | ❌ (described in report §7) |
| 5 | Photo tier + whole-property stitch | — | — | ❌ (described in report §7) |
| 6 | Per-room dimensioned plan (walls, area, outline) | `pipeline/room_outline.py`, `pipeline/room_segment.py`, `pipeline/render.py` | `floor_plan.png` | ◻ (dominant room isolated via doorway-break segmentation; multi-room split still future work) |
| 7 | Ceiling height | `pipeline/planes.py` | `result.json → room.ceiling_height` | ✅ (±1.4 cm) |
| 8 | Stitched multi-room plan with adjacency | — | — | ❌ (single capture only) |
| 9 | Per-surface damage regions + class + extent | — | — | ❌ (not in scope of this build) |
| 10 | Concealed-damage flags | — | — | ❌ |
| 11 | Scope line items keyed to surfaces | — | — | ❌ |
| 12 | Confidence interval on every measurement | `pipeline/result_schema.py` | `result.json` CIs | ✅ |
| 13 | One command per capture | `run.py` | `python run.py <scan>` | ✅ |
| 14 | JSON to a published schema | `pipeline/result_schema.py` | `result.json` (schema v1.0) | ✅ |
| 15 | Rendered plan | `pipeline/render.py` | `floor_plan.png` | ✅ |
| 16 | Opening widths (doors/windows) | — | — | ❌ (report §7) |
| 17 | Repeatability gate | — | — | ❌ (no repeat-capture pair available in sample data) |
| 18 | Drift accountability + ablation | `docs/TECHNICAL_REPORT.md §4` | Honest "poses as-is" statement | ◻ (stated, not corrected) |
| 19 | Head-to-head vs consumer app | — | — | ❌ (needs own capture + app export) |
| 20 | Fix loop (worst gate → fix → before/after) | `fixloop/FIX_LOOP.md`, `fixloop/before.txt`, `fixloop/after.txt` | FAIL→PASS, regenerable | ✅ |
| 21 | Process evidence (incremental git history) | git log | ~9 incremental commits, authored | ✅ |
| 22 | Reproduction (regenerate numbers from raw) | `run.py`, `requirements.txt`, `README.md` | Deterministic re-run | ✅ |
| 23 | Technical report | `docs/TECHNICAL_REPORT.md` | 8 sections | ✅ |
| 24 | Known failure modes (mirror/glass/wet/low light) | `docs/TECHNICAL_REPORT.md §6` | Section | ✅ |

## Summary

This build delivers a **reproducible, honestly-calibrated LiDAR-tier
pipeline** with per-room segmentation (dominant enclosed room), a complete fix
loop, and clean process evidence. It does **not** implement the photo/video
tiers, full multi-room stitching, damage detection, or drift correction. Those
are described as planned work in the technical report rather than claimed as
working — in line with the brief's emphasis on honest calibration over
confident overreach.
