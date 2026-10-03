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
| 8 | Stitched multi-room plan with adjacency | `pipeline/room_segment.py` (`count_separable_rooms`) | `result.json → capture.separable_room_count` | ◻ (detects how many rooms a capture contains; all 3 samples are single open spaces, so stitching has nothing to stitch. Multi-room split is the documented next step) |
| 9 | Per-surface damage regions + class + extent | — | — | ❌ (not in scope of this build) |
| 10 | Concealed-damage flags | — | — | ❌ |
| 11 | Scope line items keyed to surfaces | — | — | ❌ |
| 12 | Confidence interval on every measurement | `pipeline/result_schema.py` | `result.json` CIs | ✅ |
| 13 | One command per capture | `run.py` | `python run.py <scan>` | ✅ |
| 14 | JSON to a published schema | `pipeline/result_schema.py` | `result.json` (schema v1.0) | ✅ |
| 15 | Rendered plan | `pipeline/render.py` | `floor_plan.png` | ✅ |
| 15b | Benchmark summary across scans | `fixloop/benchmark_summary.py` | `docs/BENCHMARK.md` | ✅ |
| 15c | Smoke tests | `tests/test_pipeline.py` | `python -m unittest discover tests` | ✅ |
| 16 | Opening widths (doors/windows) | `pipeline/openings.py` | `result.json → room.openings`, markers on `floor_plan.png` | ◻ (detected as wall-density gaps with width CIs; conservative, can miss on jagged outlines) |
| 17 | Repeatability gate | `fixloop/repeatability.py`, `docs/REPEATABILITY.md` | Frame-split proxy: perimeter repeats to 0.04% | ◻ (proxy, not two physical captures; honestly flagged) |
| 18 | Drift accountability + ablation | `docs/TECHNICAL_REPORT.md §4` | Honest "poses as-is" statement | ◻ (stated, not corrected) |
| 19 | Head-to-head vs consumer app | — | — | ❌ (needs own capture + app export) |
| 20 | Fix loop (worst gate → fix → before/after) | `fixloop/FIX_LOOP.md`, `fixloop/before.txt`, `fixloop/after.txt` | FAIL→PASS, regenerable | ✅ |
| 21 | Process evidence (incremental git history) | git log | Incremental, authored commits built in dependency order | ✅ |
| 22 | Reproduction (regenerate numbers from raw) | `run.py`, `requirements.txt`, `README.md` | Deterministic re-run | ✅ |
| 23 | Technical report | `docs/TECHNICAL_REPORT.md` | Architecture, error budget, calibration, drift, fix loop, repeatability, segmentation, failure modes, next steps | ✅ |
| 24 | Known failure modes (mirror/glass/wet/low light) | `docs/TECHNICAL_REPORT.md §6` | Section | ✅ |

## Summary

This build delivers a **reproducible, honestly-calibrated LiDAR-tier
pipeline** with per-room segmentation (dominant enclosed room), a complete fix
loop, and clean process evidence. It does **not** implement the photo/video
tiers, full multi-room stitching, damage detection, or drift correction. Those
are described as planned work in the technical report rather than claimed as
working — in line with the brief's emphasis on honest calibration over
confident overreach.
