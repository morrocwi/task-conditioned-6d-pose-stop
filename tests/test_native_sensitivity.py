import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab.native_sensitivity import (
    NativeConfig,
    deviation_axes,
    first_act_stage,
    perturbation_magnitude,
    perturbed_pose,
    select_least_constrained_direction,
    stage_decision,
)

BIG_TASK = {"kind": "box", "mask": [0, 1, 2, 3, 4, 5], "tol": [1e6] * 6}
TINY_TASK = {"kind": "box", "mask": [0, 1, 2, 3, 4, 5], "tol": [1e-9] * 6}


class TestSelectDirection(unittest.TestCase):
    def test_picks_smallest_eigenvalue_and_matching_vector(self):
        H = np.diag([5.0, 4.0, 3.0, 2.0, 1.0, 0.5])
        lam_min, v = select_least_constrained_direction(H)
        self.assertAlmostEqual(lam_min, 0.5, places=9)
        # eigenvector for a diagonal matrix's smallest eigenvalue is +-e_5
        expected = np.zeros(6)
        expected[5] = 1.0
        self.assertTrue(np.allclose(np.abs(v), expected, atol=1e-9))

    def test_symmetrizes_roundoff(self):
        H = np.eye(6) + 1e-14 * np.arange(36).reshape(6, 6)
        lam_min, v = select_least_constrained_direction(H)
        self.assertTrue(math.isfinite(lam_min))
        self.assertAlmostEqual(np.linalg.norm(v), 1.0, places=9)


class TestPerturbationMagnitude(unittest.TestCase):
    def setUp(self):
        self.cfg = NativeConfig(lambda_ref=0.34, cap=0.005)

    def test_saturates_at_cap_for_small_eigenvalue(self):
        m, degenerate = perturbation_magnitude(1e-6, self.cfg)
        self.assertFalse(degenerate)
        self.assertAlmostEqual(m, self.cfg.cap, places=12)

    def test_shrinks_below_cap_for_large_eigenvalue(self):
        m, degenerate = perturbation_magnitude(1000.0, self.cfg)
        self.assertFalse(degenerate)
        self.assertLess(m, self.cfg.cap)
        self.assertGreater(m, 0.0)

    def test_fail_closed_to_cap_on_singular_H(self):
        m, degenerate = perturbation_magnitude(0.0, self.cfg)
        self.assertTrue(degenerate)
        self.assertEqual(m, self.cfg.cap)

    def test_fail_closed_to_cap_on_negative_or_nonfinite(self):
        for bad in (-1.0, math.nan, math.inf):
            m, degenerate = perturbation_magnitude(bad, self.cfg)
            self.assertTrue(degenerate)
            self.assertEqual(m, self.cfg.cap)

    def test_never_negative_never_exceeds_cap(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            lam = float(rng.uniform(-1.0, 1000.0))
            m, _ = perturbation_magnitude(lam, self.cfg)
            self.assertGreaterEqual(m, 0.0)
            self.assertLessEqual(m, self.cfg.cap + 1e-15)


class TestDeviationAndPerturbedPose(unittest.TestCase):
    def test_zero_perturbation_gives_zero_deviation(self):
        R = np.eye(3)
        t = np.array([0.1, 0.2, 0.3])
        dev = deviation_axes(R, t, R, t)
        self.assertTrue(np.allclose(dev, 0.0, atol=1e-12))

    def test_nonzero_perturbation_gives_nonzero_deviation(self):
        R = np.eye(3)
        t = np.array([0.0, 0.0, 0.0])
        v = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])  # pure +tx direction
        R2, t2 = perturbed_pose(R, t, v, 0.01, +1.0)
        dev = deviation_axes(R, t, R2, t2)
        self.assertAlmostEqual(dev[0], 0.01, places=9)
        self.assertTrue(np.allclose(dev[1:], 0.0, atol=1e-9))

    def test_opposite_signs_give_opposite_but_equal_magnitude_deviation(self):
        R = np.eye(3)
        t = np.zeros(3)
        v = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # pure rotation direction
        Rp, tp = perturbed_pose(R, t, v, 0.02, +1.0)
        Rm, tm = perturbed_pose(R, t, v, 0.02, -1.0)
        dev_p = deviation_axes(R, t, Rp, tp)
        dev_m = deviation_axes(R, t, Rm, tm)
        self.assertAlmostEqual(dev_p[3], dev_m[3], places=6)


class TestStageDecisionAndActLogic(unittest.TestCase):
    def setUp(self):
        # A poorly-conditioned H (small lambda_min) so CAP saturates and the
        # perturbation is not negligible.
        self.H = np.diag([500.0, 500.0, 500.0, 500.0, 500.0, 0.3])
        self.cfg = NativeConfig(lambda_ref=0.34, cap=0.005)
        self.R = np.eye(3)
        self.t = np.zeros(3)

    def test_acts_when_task_tolerance_is_huge(self):
        d = stage_decision(self.R, self.t, self.H, BIG_TASK, self.cfg)
        self.assertTrue(d["verdict_baseline"])
        self.assertTrue(d["verdict_plus"])
        self.assertTrue(d["verdict_minus"])
        self.assertTrue(d["act"])

    def test_holds_when_task_tolerance_is_tiny(self):
        d = stage_decision(self.R, self.t, self.H, TINY_TASK, self.cfg)
        self.assertTrue(d["verdict_baseline"])  # zero deviation always passes
        self.assertFalse(d["verdict_plus"])
        self.assertFalse(d["verdict_minus"])
        self.assertFalse(d["act"])

    def test_first_act_stage_finds_first_qualifying_stage(self):
        stages = [
            {"native": {"hessian": [float(x) for x in np.diag([500, 500, 500, 500, 500, 0.001]).reshape(-1)],
                        "R": self.R.tolist(), "t": self.t.tolist()}},
            {"native": {"hessian": [float(x) for x in np.diag([500, 500, 500, 500, 500, 5000]).reshape(-1)],
                        "R": self.R.tolist(), "t": self.t.tolist()}},
        ]
        idx, acted, audit = first_act_stage(stages, BIG_TASK, self.cfg)
        self.assertEqual(len(audit), 1)  # stopped at first qualifying stage
        self.assertEqual(idx, 0)
        self.assertTrue(acted)

    def test_first_act_stage_none_when_never_qualifies(self):
        stages = [
            {"native": {"hessian": [float(x) for x in np.diag([500, 500, 500, 500, 500, 0.001]).reshape(-1)],
                        "R": self.R.tolist(), "t": self.t.tolist()}},
        ]
        idx, acted, audit = first_act_stage(stages, TINY_TASK, self.cfg)
        self.assertIsNone(idx)
        self.assertFalse(acted)
        self.assertEqual(len(audit), 1)


if __name__ == "__main__":
    unittest.main()
