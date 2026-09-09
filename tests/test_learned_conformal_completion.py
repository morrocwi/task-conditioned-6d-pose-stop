import inspect
import math
import unittest

import numpy as np

from cqts.safety import CertificateNumericsError
from experiments import learned_conformal_completion as lc
from experiments import numerical_icp as ni


class TestLearnedConformalCompletion(unittest.TestCase):
    def test_gate_signature_excludes_ground_truth(self):
        params = tuple(inspect.signature(lc.first_completion_stage).parameters)
        self.assertEqual(params, ("stages", "task", "model", "q"))
        for forbidden in ("ep", "Rg", "tg", "ground_truth"):
            self.assertNotIn(forbidden, params)

    def test_train_cal_final_seeds_are_disjoint(self):
        values = {lc.TRAIN_SEED, lc.CAL_SEED, lc.STRESS_SEED, *lc.FINAL_TEST_SEEDS}
        self.assertEqual(len(values), 3 + len(lc.FINAL_TEST_SEEDS))
        self.assertTrue({2026090931, 2026090932, 2026090933}.isdisjoint(values))

    def test_split_conformal_rank(self):
        q, rank = lc.split_conformal_quantile(list(range(1, 10)), alpha=0.2)
        self.assertEqual(rank, 8)
        self.assertEqual(q, 8.0)

    def test_small_calibration_uses_augmented_infinity(self):
        q, rank = lc.split_conformal_quantile([1.0, 2.0], alpha=0.1)
        self.assertEqual(rank, 3)
        self.assertEqual(q, math.inf)

    def test_nonfinite_calibration_is_rejected(self):
        with self.assertRaises(CertificateNumericsError):
            lc.split_conformal_quantile([1.0, math.nan], alpha=0.1)

    def test_invalid_model_fails_closed_not_zero_width(self):
        rng = np.random.default_rng(44)
        ep = ni.generate_episode(rng, max_iter=2, points=24)
        bad = lc.ShapeModel(beta=np.full((6, 5), math.nan), max_iter=2)
        b = lc.completion_bounds(ep.stages[0], bad, 1.0)
        self.assertTrue(np.all(np.isinf(b)))
        self.assertFalse(lc.robust_task_pass("keyed_insertion", b))

    def test_unbounded_calibration_holds(self):
        rng = np.random.default_rng(45)
        ep = ni.generate_episode(rng, max_iter=2, points=24)
        model = lc.fit_shape_model([ep], 2)
        k, act, _ = lc.first_completion_stage(ep.stages, "top_suction", model, math.inf)
        self.assertIsNone(k)
        self.assertFalse(act)

    def test_robust_pass_implies_success_for_any_error_inside_box(self):
        rng = np.random.default_rng(456)
        examples = {
            "top_suction": np.array([0.003, 0.003, 0.005, math.radians(3), math.radians(3), math.radians(20)]),
            "label_alignment": np.array([0.003, 0.003, 0.005, math.radians(3), math.radians(3), math.radians(2)]),
            "keyed_insertion": np.array([0.001, 0.001, 0.0, 0.0, 0.0, math.radians(0.5)]),
        }
        for task, bounds in examples.items():
            self.assertTrue(lc.robust_task_pass(task, bounds))
            for _ in range(100):
                err = rng.uniform(-1.0, 1.0, size=6) * bounds
                self.assertTrue(ni.task_success(task, err))

    def test_ci_pipeline_schema_and_covered_act_implication(self):
        out = lc.run("ci", "artifacts/test-learned-conformal")
        self.assertEqual(out["schema_version"], 2)
        self.assertFalse(out["evidence_boundary"]["hidden_ground_truth_used_online_by_gate"])
        self.assertFalse(out["evidence_boundary"]["trajectory_prefix_timing_is_online_speedup"])
        for condition in out["evaluation"].values():
            self.assertGreaterEqual(condition["trajectory_envelope_coverage"], 0.0)
            self.assertLessEqual(condition["trajectory_envelope_coverage"], 1.0)
            for task in condition["tasks"].values():
                self.assertEqual(task["unsafe_act_on_covered_episode_count"], 0)


if __name__ == "__main__":
    unittest.main()
