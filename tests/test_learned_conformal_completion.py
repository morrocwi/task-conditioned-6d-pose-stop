import inspect
import math
import unittest

import numpy as np

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
        old_dev = {2026090931, 2026090932, 2026090933}
        self.assertTrue(old_dev.isdisjoint(values))

    def test_split_conformal_rank(self):
        q, rank = lc.split_conformal_quantile(list(range(1, 10)), alpha=0.2)
        self.assertEqual(rank, 8)
        self.assertEqual(q, 8.0)

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
        self.assertEqual(out["label"], "[NumericalBackend+LearnedShape+SplitConformal]")
        self.assertFalse(out["evidence_boundary"]["hidden_ground_truth_used_online_by_gate"])
        for condition in out["evaluation"].values():
            self.assertGreaterEqual(condition["trajectory_envelope_coverage"], 0.0)
            self.assertLessEqual(condition["trajectory_envelope_coverage"], 1.0)
            for task in condition["tasks"].values():
                self.assertEqual(task["unsafe_act_on_covered_episode_count"], 0)


if __name__ == "__main__":
    unittest.main()
