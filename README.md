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

## Scope (honest)

This build implements and verifies the **LiDAR tier** only:
reliable ceiling height (±1.4 cm), floor area, extents, and orientation, each
with a calibrated confidence interval, plus a rendered plan and JSON output.

Not implemented (described as planned work in the technical report): photo and
video tiers, multi-room stitching, per-room segmentation, openings detection,
damage detection, and drift correction. The room outline is currently the
scanned footprint envelope, which can include regions seen through open
doorways.

## Reproduce a result

```bash
python run.py sample_data/single_scan_with_ceiling/c7d28f72c6
python -m fixloop.measure_ceiling outputs/c7d28f72c6/cloud.ply   # fix-loop number
```
