# Capture Protocol (Route 2: stock app)

This pipeline consumes the **StrayScanner** export format, so capture uses an
off-the-shelf App Store app and a one-page protocol a non-engineer can follow.
No custom iOS build is required.

## App

- **App:** Stray Scanner (App Store, free), the same format as the provided
  sample data.
- **Export:** the app writes a folder per scan containing `rgb.mp4`,
  `depth/`, `confidence/`, `odometry.csv`, `camera_matrix.csv`, `imu.csv`.

## One-page protocol (hand this to the capturer)

1. **Install** "Stray Scanner" from the App Store on an iPhone/iPad **Pro**
   (LiDAR required — see device matrix).
2. **Stand in a corner** of the room, phone held at chest height, screen
   facing you, rear camera facing the wall.
3. **Tap record.** Walk the **perimeter slowly** (about one step per second),
   keeping the far wall in view. Sweep the phone gently up and down so both
   the **floor and the ceiling** are seen.
4. **Keep going** until you return to your starting corner, then do **one
   extra lap**. A full room takes about **60-90 seconds**.
5. **Avoid**: pointing straight into mirrors or large glass, standing still
   for long, and fast whips (they blur poses). Keep the phone 0.5-4 m from
   whatever it is looking at.
6. **Stop**, then **export/AirDrop** the scan folder to the machine running
   the pipeline. Hand it over as the folder (not re-zipped).

Then run, on that machine:

```bash
python run.py path/to/<scan_folder>
```

## Device matrix

| Tier   | Hardware                              | What it delivers | Honest accuracy (this build) |
|--------|---------------------------------------|------------------|------------------------------|
| LiDAR  | iPhone 12 Pro and newer "Pro" models, iPad Pro | Depth + poses + intrinsics | Ceiling height ±1.4 cm; floor area with calibrated CI. **Implemented.** |
| Video  | Any iPhone 15+                        | RGB walkthrough, no depth | Not implemented in this build; would need monocular SLAM + scale. |
| Photo  | Any iPhone 15+                        | 2-8 stills per room | Not implemented in this build; would need multi-view/learned depth. |

This submission implements and verifies the **LiDAR tier** only. The video and
photo tiers are described in the technical report as the planned extension but
are **not** claimed to work here — see `docs/TECHNICAL_REPORT.md`.
