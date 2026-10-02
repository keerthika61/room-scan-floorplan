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

    def test_floor_only_reports_no_ceiling(self):
        pts = self._synthetic_room(with_ceiling=False)
        res = find_floor_ceiling(pts)
        self.assertIsNone(res.ceiling)
        self.assertIsNone(res.ceiling_height_m)


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
        for w in room["walls"]:
            self.assertIn("ci_half_width", w["length"])
        self.assertEqual(room["wall_count"], 4)


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
