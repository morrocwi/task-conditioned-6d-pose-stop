import importlib.util
import inspect
import math
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("labrun", ROOT / "lab" / "run_real_system.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class TestLabHarness(unittest.TestCase):
    def test_gate_has_no_oracle_argument(self):
        params = set(inspect.signature(m.first_certificate_stage).parameters)
        self.assertEqual(params, {"stages", "spec", "model", "q"})

    def test_split_leakage_rejected(self):
        row = {"episode_id": "x", "stages": [{"k": 0, "features": [1.0], "incremental_ms": 1.0, "estimator_stop": True}], "oracle": {"abs_pose_error_6d": [[0, 0, 0, 0, 0, 0]]}}
        with self.assertRaises(ValueError):
            m.ensure_disjoint(("train", [row]), ("test", [row]))

    def test_conformal_quantile(self):
        q, rank = m.conformal_quantile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.2)
        self.assertEqual(rank, 9)
        self.assertEqual(q, 9)

    def test_small_calibration_becomes_unbounded(self):
        q, rank = m.conformal_quantile([1.0, 2.0], 0.1)
        self.assertEqual(rank, 3)
        self.assertEqual(q, math.inf)

    def test_robust_l1_is_joint(self):
        task = {"kind": "l1", "mask": [0, 1, 5], "tol": [1, 1, 99, 99, 99, 1]}
        self.assertFalse(m.task_pass(task, [0.4, 0.4, 0, 0, 0, 0.4]))
        self.assertTrue(m.task_pass(task, [0.2, 0.2, 0, 0, 0, 0.2]))

    def test_malformed_tolerance_fails_closed(self):
        task = {"kind": "box", "mask": [0], "tol": [0.0, 1, 1, 1, 1, 1]}
        self.assertFalse(m.task_pass(task, [0, 0, 0, 0, 0, 0]))

    def test_invalid_model_produces_unbounded_completion(self):
        model = m.Model(beta=np.full((6, 2), math.nan), mean=np.array([0.0]), scale=np.array([1.0]), error_floor=np.ones(6) * 1e-5)
        b = m.completion_bounds({"features": [0.0]}, model, 1.0)
        self.assertTrue(np.all(np.isinf(b)))

    def test_one_episode_equal_outcomes_do_not_certify_noninferiority(self):
        out = m.paired_binary_noninferiority([1], [1], margin=0.05, confidence_level=0.95, minimum_n=1)
        self.assertFalse(out["noninferiority_pass"])
        self.assertLess(out["lower_confidence_bound"], -0.05)

    def test_predeclared_minimum_sample_is_enforced(self):
        out = m.paired_binary_noninferiority([1] * 20, [1] * 20, margin=0.05, confidence_level=0.95, minimum_n=30)
        self.assertFalse(out["sufficient_sample_for_declared_plan"])
        self.assertFalse(out["noninferiority_pass"])


if __name__ == "__main__":
    unittest.main()
