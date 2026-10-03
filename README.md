# Room Scan → Floor Plan Pipeline

Turns an iPhone LiDAR scan (depth + camera poses) into a dimensioned floor plan
with measurements and calibrated confidence intervals.

Built for the Applied AI Engineer case study. Current focus: the **LiDAR tier**.

## What it does

Given one scan folder, the pipeline produces:

- A fused 3D point cloud of the room
- Ceiling height, wall lengths, floor area
- Door/window openings and their widths
- A confidence interval on every measurement
- A JSON result file (to a fixed schema) and a rendered top-down floor plan

## Validated accuracy (vs. synthetic ground truth)

Each core measurement is checked against inputs with known answers
(`python -m unittest discover tests`, 18 tests):

| Measurement | Ground-truth test | Result |
|-------------|-------------------|--------|
| Ceiling height | known 2.40 / 2.70 / 3.00 m rooms (+ fixture clutter) | < 0.5 mm error, true value inside CI |
| Floor area | known 4×3 m room (12 m²) | 0.5 % error |
| Orientation | rooms rotated 0–60° | exact (0°) |
| Opening width | known 0.90 m doorway | exact (0.90 m) |
| Determinism | same cloud, repeated runs | bit-identical |

On real data, the fused floor is flat to ~2 cm, confirming the pose /
back-projection math.

## Input format

Each scan is a folder (StrayScanner-style iPhone LiDAR export):

```
<scan>/
  rgb.mp4              # walkthrough video (1920x1440)
  depth/000000.png ... # per-frame 16-bit depth in millimeters (256x192)
  confidence/000000.png# per-frame LiDAR confidence 0/1/2 (256x192)
  odometry.csv         # per-frame camera pose: x,y,z + quaternion qx,qy,qz,qw
  camera_matrix.csv    # 3x3 camera intrinsics (for the RGB resolution)
  imu.csv              # raw IMU (unused in the core LiDAR path)
```

## Repo map

```
run.py                     single-command entry point
pipeline/
  scan_reader.py           parse poses, intrinsics (scaled to depth res), frames
  pointcloud.py            back-project depth + poses -> fused world point cloud
  planes.py                floor/ceiling via robust plane fit -> ceiling height + CI
  room_segment.py          isolate dominant room; count separable rooms
  room_outline.py          orientation (minAreaRect) + simplified outline, areas
  openings.py              door/window widths from wall-density gaps
  video.py                 rgb.mp4 ingestion + frame-alignment check
  colorize.py              RGB-colored point cloud from aligned rgb.mp4 (--colorize)
  anomaly.py               unsupervised surface-anomaly indicator (damage scaffold)
  render.py                top-down floor plan PNG
  result_schema.py         JSON result (value + CI + method on every measurement)
tests/test_pipeline.py     13 tests incl. ground-truth accuracy checks
fixloop/                   fix loop (FAIL->PASS), repeatability, benchmark tools
docs/                      report, compliance matrix, capture protocol, benchmark
```

## Setup

```bash
python -m pip install -r requirements.txt
```

Place the sample scans under `sample_data/` (not committed; large).

## Run (one command per capture)

```bash
python run.py sample_data/single_room/c00a170fe1
```

Outputs are written to `outputs/<scan_name>/`.

## Documents

- `docs/TECHNICAL_REPORT.md` — architecture, error budget, calibration, drift status, failure modes, next steps.
- `docs/COMPLIANCE_MATRIX.md` — requirement → file → artifact → status (honest).
- `docs/CAPTURE_PROTOCOL.md` — Route 2 stock-app capture protocol + device matrix.
- `fixloop/FIX_LOOP.md` — the measure → diagnose → fix → re-measure cycle (FAIL→PASS), with regenerable before/after.
- `docs/REPEATABILITY.md` — repeatability proxy (perimeter repeats to 0.04%).
- `docs/BENCHMARK.md` — measurements across all three scans.
- `docs/SUBMISSION_CHECKLIST.md` — what to run and where each deliverable lives.

## Scope (honest)

This build implements and verifies the **LiDAR tier** only:
reliable ceiling height (±1.4 cm), floor area, extents, and orientation, each
with a calibrated confidence interval, plus a rendered plan and JSON output.

Per-room segmentation isolates the dominant enclosed room (breaking narrow
doorway necks), with an honest fallback to the full footprint when no single
compact room dominates. Doors/windows are detected as gaps in wall-height
density and reported with widths + confidence intervals.

Not implemented (described as planned work in the technical report): photo and
video tiers, full multi-room stitching, damage detection, and drift
correction.

## Tests and benchmark

```bash
python -m unittest discover tests          # fast smoke tests
python -m fixloop.benchmark_summary        # regenerate docs/BENCHMARK.md
```

## Reproduce a result

```bash
python run.py sample_data/single_scan_with_ceiling/c7d28f72c6
python -m fixloop.measure_ceiling outputs/c7d28f72c6/cloud.ply   # fix-loop number
```
