# Submission Checklist

A reviewer should be able to go from clone to a reproduced result in a few
minutes. This lists what to run and what each deliverable is.

## Run it

```bash
python -m pip install -r requirements.txt
# place the sample scans under sample_data/ (not committed; large)
python run.py sample_data/single_room/c00a170fe1
```

Outputs land in `outputs/<scan>/`: `cloud.ply`, `result.json`, `floor_plan.png`.

## Verify it

```bash
python -m unittest discover tests            # 7 fast tests, all pass
python -m fixloop.measure_ceiling outputs/c7d28f72c6/cloud.ply   # fix-loop number (PASS)
python -m fixloop.repeatability sample_data/single_room/c00a170fe1
python -m fixloop.benchmark_summary          # regenerates docs/BENCHMARK.md
```

## Deliverables map

| Deliverable | Where |
|-------------|-------|
| Code (pipeline) | `pipeline/`, `run.py` |
| Single-command run | `python run.py <scan>` |
| JSON result (CI on every measurement) | `outputs/<scan>/result.json` |
| Rendered plan | `outputs/<scan>/floor_plan.png` |
| Fix loop (FAIL->PASS) | `fixloop/FIX_LOOP.md` (+ before/after) |
| Repeatability | `docs/REPEATABILITY.md` |
| Benchmark across scans | `docs/BENCHMARK.md` |
| Technical report | `docs/TECHNICAL_REPORT.md` |
| Compliance matrix | `docs/COMPLIANCE_MATRIX.md` |
| Capture protocol + device matrix | `docs/CAPTURE_PROTOCOL.md` |
| Tests | `tests/test_pipeline.py` |

## Honest scope

Implemented and verified: **LiDAR tier** — ceiling height (±1.4 cm), floor
area, free-form room outline + wall lengths, per-room segmentation,
door/window openings, calibrated confidence intervals, a fix loop, and a
repeatability proxy.

Not implemented (stated as future work in the report): photo/video tiers, full
multi-room stitching, damage detection, and independent drift correction.
