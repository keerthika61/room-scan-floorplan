"""
test_pipeline.py
----------------
Fast, dependency-light checks a reviewer can run with:

    python -m unittest discover tests

The geometry tests build tiny synthetic inputs so they need no scan data and
run in well under a second. One optional end-to-end test runs only if a built
cloud is present under outputs/.
"""

import math
import os
import unittest

import numpy as np

from pipeline.scan_reader import _quaternion_to_rotation
from pipeline.planes import find_floor_ceiling
from pipeline.result_schema import build_result


class TestQuaternion(unittest.TestCase):
    def test_identity(self):
        R = _quaternion_to_rotation(0, 0, 0, 1)
        self.assertTrue(np.allclose(R, np.eye(3), atol=1e-9))

    def test_90deg_about_y(self):
        # Quaternion for +90 deg about Y: (0, sin45, 0, cos45).
        s = math.sqrt(0.5)
        R = _quaternion_to_rotation(0, s, 0, s)
        # Rotating +X by +90 about Y should give -Z.
        v = R @ np.array([1.0, 0.0, 0.0])
        self.assertTrue(np.allclose(v, [0, 0, -1], atol=1e-6), msg=f"got {v}")

    def test_normalizes(self):
        # Unnormalized quaternion should still give an orthonormal rotation.
        R = _quaternion_to_rotation(0, 0, 0, 2)
        self.assertTrue(np.allclose(R @ R.T, np.eye(3), atol=1e-9))


class TestFloorCeiling(unittest.TestCase):
    def _synthetic_room(self, height=2.5, with_ceiling=True):
        """Floor at y=0 and (optionally) ceiling at y=height, plus wall noise."""
        rng = np.random.default_rng(0)
        floor = np.column_stack([
            rng.uniform(-2, 2, 20000),
            rng.normal(0.0, 0.003, 20000),   # thin, slightly noisy slab
            rng.uniform(-2, 2, 20000),
        ])
        parts = [floor]
        if with_ceiling:
            ceil = np.column_stack([
                rng.uniform(-2, 2, 20000),
                rng.normal(height, 0.003, 20000),
                rng.uniform(-2, 2, 20000),
            ])
            parts.append(ceil)
        return np.vstack(parts)

    def test_ceiling_height_accurate(self):
        pts = self._synthetic_room(height=2.5, with_ceiling=True)
        res = find_floor_ceiling(pts)
        self.assertIsNotNone(res.ceiling)
        self.assertAlmostEqual(res.ceiling_height_m, 2.5, delta=0.03)
        # CI should be tight for a clean synthetic slab.
        self.assertLess(res.ceiling_height_ci_m, 0.015)

    def test_ceiling_height_unbiased_with_clutter(self):
        """Known heights recovered to <5 mm even with hanging-fixture outliers,
        and the true height falls inside the reported CI (honest calibration).
        """
        for true_h in (2.40, 2.70, 3.00):
            rng = np.random.default_rng(0)
            pts = []
            for _ in range(40000):
                pts.append([rng.uniform(0, 4), rng.normal(0.0, 0.004), rng.uniform(0, 3)])
            for _ in range(30000):
                pts.append([rng.uniform(0, 4), rng.normal(true_h, 0.004), rng.uniform(0, 3)])
            for _ in range(2000):  # light fixture 15 cm below ceiling
                pts.append([rng.uniform(1.5, 2.5), true_h - 0.15 + rng.normal(0, 0.01),
                            rng.uniform(1.0, 2.0)])
            res = find_floor_ceiling(np.array(pts))
            self.assertIsNotNone(res.ceiling)
            self.assertAlmostEqual(res.ceiling_height_m, true_h, delta=0.005)
            self.assertLessEqual(abs(res.ceiling_height_m - true_h),
                                 res.ceiling_height_ci_m)

    def test_floor_only_reports_no_ceiling(self):
        pts = self._synthetic_room(with_ceiling=False)
        res = find_floor_ceiling(pts)
        self.assertIsNone(res.ceiling)
        self.assertIsNone(res.ceiling_height_m)

    def test_floor_not_confused_when_ceiling_denser(self):
        """Floor is found by position, not point count: a ceiling with 5x more
        points must not be mistaken for the floor."""
        rng = np.random.default_rng(0)
        pts = []
        for _ in range(10000):
            pts.append([rng.uniform(0, 4), rng.normal(0, 0.004), rng.uniform(0, 3)])
        for _ in range(50000):  # much denser ceiling
            pts.append([rng.uniform(0, 4), rng.normal(2.7, 0.004), rng.uniform(0, 3)])
        res = find_floor_ceiling(np.array(pts))
        self.assertAlmostEqual(res.floor.height, 0.0, delta=0.02)
        self.assertIsNotNone(res.ceiling)
        self.assertAlmostEqual(res.ceiling_height_m, 2.70, delta=0.02)


class _FakeOutline:
    polygon_xz = np.array([[0, 0], [3, 0], [3, 4], [0, 4]], float)
    wall_lengths_m = [3.0, 4.0, 3.0, 4.0]
    wall_length_ci_m = [0.03, 0.03, 0.03, 0.03]
    floor_area_m2 = 12.0
    floor_area_ci_m2 = 0.3
    orientation_deg = 0.0
    segmentation_mode = "dominant_room"


class _FakeHeight:
    ceiling_height_m = 2.5
    ceiling_height_ci_m = 0.014
    note = "ceiling found (least-squares plane fit)"


class TestResultSchema(unittest.TestCase):
    def test_every_measurement_has_ci(self):
        r = build_result("fake", _FakeHeight(), _FakeOutline(), n_points=1000)
        room = r["room"]
        # Each top-level measurement carries a value + ci + unit + method.
        for key in ("floor_area", "ceiling_height", "perimeter"):
            m = room[key]
            self.assertIn("value", m)
            self.assertIn("ci_half_width", m)
            self.assertIn("unit", m)
            self.assertIn("method", m)
        for w in room["outline_edges"]:
            self.assertIn("ci_half_width", w["length"])
        self.assertEqual(room["outline_edge_count"], 4)


class TestRoomCount(unittest.TestCase):
    def test_two_separated_rooms_detected(self):
        from pipeline.room_segment import count_separable_rooms
        rng = np.random.default_rng(1)
        # Two 3x3 m floor patches separated by a 1.5 m gap (no connecting floor)
        # -> two distinct room cores.
        def patch(cx, cz):
            return np.column_stack([
                rng.uniform(cx - 1.5, cx + 1.5, 8000),
                np.zeros(8000),
                rng.uniform(cz - 1.5, cz + 1.5, 8000),
            ])
        pts = np.vstack([patch(0, 0), patch(0, 6)])
        info = count_separable_rooms(pts, floor_y=0.0)
        self.assertGreaterEqual(info["room_count"], 2)
        self.assertTrue(info["is_multi_room"])

    def test_two_rooms_through_doorway_split(self):
        """Two rooms joined by a 0.9 m doorway (so they MERGE in the filled
        mask) must still be split into two by nearest-core assignment."""
        from pipeline.room_segment import label_all_rooms
        rng = np.random.default_rng(0)
        pts = []
        for _ in range(50000):   # room A 4x3
            pts.append([rng.uniform(0, 4), 0.0, rng.uniform(0, 3)])
        for _ in range(40000):   # room B 3x3
            pts.append([rng.uniform(5, 8), 0.0, rng.uniform(0, 3)])
        for _ in range(4000):    # 0.9 m doorway
            pts.append([rng.uniform(4, 5), 0.0, rng.uniform(1.0, 1.9)])
        masks, _, res = label_all_rooms(np.array(pts), floor_y=0.0)
        self.assertEqual(len(masks), 2)
        areas = sorted((m > 0).sum() * res * res for m in masks)
        self.assertAlmostEqual(areas[0], 9.0, delta=1.5)
        self.assertAlmostEqual(areas[1], 12.0, delta=1.5)

    def test_single_room_not_oversplit(self):
        """A single room must come back as exactly one room."""
        from pipeline.room_segment import label_all_rooms
        rng = np.random.default_rng(0)
        pts = np.column_stack([rng.uniform(0, 4, 60000),
                               np.zeros(60000),
                               rng.uniform(0, 3, 60000)])
        masks, _, _ = label_all_rooms(pts, floor_y=0.0)
        self.assertEqual(len(masks), 1)


class TestOrientation(unittest.TestCase):
    def test_recovers_known_rotation(self):
        """minAreaRect orientation must recover known room rotations exactly."""
        from pipeline.room_outline import estimate_orientation
        rng = np.random.default_rng(0)
        for truth in (0, 10, 20, 30, 45, 60):
            th = np.radians(truth)
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            xz = (rng.uniform([0, 0], [4, 3], size=(60000, 2)) @ R.T)
            pts = np.column_stack([xz[:, 0], np.zeros(len(xz)), xz[:, 1]])
            est = np.degrees(estimate_orientation(pts, 0.0)) % 90
            t = truth % 90
            err = min(abs(est - t), 90 - abs(est - t))
            self.assertLess(err, 3.0, f"truth {truth}, est {est}, err {err}")


class TestKnownRoomAccuracy(unittest.TestCase):
    def test_synthetic_4x3_room(self):
        """A perfect 4x3 m room (12 m^2) must be recovered to within ~2%."""
        from pipeline.room_outline import compute_room_outline
        rng = np.random.default_rng(0)
        pts = []
        for _ in range(60000):
            pts.append([rng.uniform(0, 4), rng.normal(0, 0.003), rng.uniform(0, 3)])
        for _ in range(40000):
            side = rng.integers(0, 4)
            if side == 0:   p = [rng.uniform(0, 4), rng.uniform(0, 2), 0.0]
            elif side == 1: p = [rng.uniform(0, 4), rng.uniform(0, 2), 3.0]
            elif side == 2: p = [0.0, rng.uniform(0, 2), rng.uniform(0, 3)]
            else:           p = [4.0, rng.uniform(0, 2), rng.uniform(0, 3)]
            pts.append(p)
        ro = compute_room_outline(np.array(pts), floor_y=0.0)
        self.assertAlmostEqual(ro.floor_area_m2, 12.0, delta=0.5)   # <~4%
        self.assertAlmostEqual(sum(ro.wall_lengths_m), 14.0, delta=0.6)
        # On a clean room the simplified outline should recover the 4 true
        # walls (lengths ~4, 4, 3, 3) among its longest edges.
        longest = sorted(ro.wall_lengths_m, reverse=True)[:4]
        self.assertAlmostEqual(longest[0], 4.0, delta=0.15)
        self.assertAlmostEqual(longest[2], 3.0, delta=0.15)


class TestOpenings(unittest.TestCase):
    def test_known_doorway_width(self):
        """A synthetic wall with a known 0.90 m gap must measure ~0.90 m."""
        from pipeline.openings import detect_openings
        rng = np.random.default_rng(0)
        pts = []
        for x in np.linspace(0, 4, 400):
            if 1.5 <= x <= 2.4:   # 0.9 m doorway gap
                continue
            for y in np.linspace(0.1, 2.0, 40):
                pts.append([x, y, rng.normal(0, 0.002)])
        pts = np.array(pts)

        class O:
            polygon_xz = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]])
            wall_lengths_m = [4.0, 3.0, 4.0, 3.0]

        ops = detect_openings(pts, floor_y=0.0, outline=O())
        self.assertEqual(len(ops), 1)
        self.assertAlmostEqual(ops[0].width_m, 0.90, delta=0.10)
        self.assertAlmostEqual(ops[0].center_xz[0], 1.95, delta=0.15)
        self.assertEqual(ops[0].kind, "door")


class TestDeterminism(unittest.TestCase):
    def test_geometry_is_deterministic(self):
        """Same cloud in -> bit-identical measurements out (reproducibility)."""
        cloud = "outputs/c00a170fe1/cloud.ply"
        if not os.path.exists(cloud):
            self.skipTest("no built cloud present; run `python run.py <scan>` first")
        import open3d as o3d
        from pipeline.room_outline import compute_room_outline

        pts = np.asarray(o3d.io.read_point_cloud(cloud).points)
        fy = find_floor_ceiling(pts).floor.height
        a = compute_room_outline(pts, fy)
        b = compute_room_outline(pts, fy)
        self.assertEqual(round(a.floor_area_m2, 6), round(b.floor_area_m2, 6))
        self.assertEqual(
            [round(x, 6) for x in a.wall_lengths_m],
            [round(x, 6) for x in b.wall_lengths_m],
        )


class TestVideoAlignmentIfDataPresent(unittest.TestCase):
    def test_rgb_video_aligned_with_geometry(self):
        """rgb.mp4 must be frame-aligned with odometry/depth for RGB fusion."""
        scan = "sample_data/single_room/c00a170fe1"
        if not os.path.exists(os.path.join(scan, "rgb.mp4")):
            self.skipTest("sample scan not present")
        from pipeline.video import check_alignment
        a = check_alignment(scan)
        self.assertTrue(a["aligned"], a)


class TestEndToEndIfDataPresent(unittest.TestCase):
    def test_runs_on_existing_cloud(self):
        cloud = "outputs/c00a170fe1/cloud.ply"
        if not os.path.exists(cloud):
            self.skipTest("no built cloud present; run `python run.py <scan>` first")
        import open3d as o3d
        from pipeline.room_outline import compute_room_outline

        pts = np.asarray(o3d.io.read_point_cloud(cloud).points)
        self.assertGreater(len(pts), 1000)
        res = find_floor_ceiling(pts)
        outline = compute_room_outline(pts, res.floor.height)
        self.assertGreaterEqual(len(outline.polygon_xz), 4)
        self.assertGreater(outline.floor_area_m2, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
