import inspect
import math
import unittest

import numpy as np

from cqts.safety import CertificateNumericsError
from experiments import conformal_completion as cc
from experiments import numerical_icp as ni


class TestConformalCompletion(unittest.TestCase):
    def test_order_statistic(self):
        q, rank = cc.split_conformal_quantile([1, 2, 3, 4, 5, 6, 7, 8, 9], alpha=0.2)
        self.assertEqual(rank, 8)
        self.assertEqual(q, 8.0)

    def test_small_n_augments_with_infinity(self):
        q, rank = cc.split_conformal_quantile([1.0, 2.0], alpha=0.1)
        self.assertEqual(rank, 3)
        self.assertEqual(q, math.inf)

    def test_nonfinite_score_is_rejected(self):
        with self.assertRaises(CertificateNumericsError):
            cc.split_conformal_quantile([1.0, math.nan], alpha=0.1)

    def test_gate_signature_has_no_episode_or_ground_truth(self):
        params = tuple(inspect.signature(cc.first_completion_stage).parameters)
        self.assertEqual(params, ("stages", "task", "q"))
        self.assertNotIn("ep", params)
        self.assertNotIn("Rg", params)
        self.assertNotIn("tg", params)

    def test_task_reader_ignores_irrelevant_yaw(self):
        wide_yaw = np.array([0.002, 0.002, 0.004, math.radians(2), math.radians(2), math.radians(12)])
        self.assertTrue(cc.robust_task_pass("top_suction", wide_yaw))
        self.assertFalse(cc.robust_task_pass("label_alignment", wide_yaw))

    def test_coupled_insertion_is_not_coordinatewise(self):
        b = np.array([0.003, 0.003, 0.0, 0.0, 0.0, math.radians(2.4)])
        self.assertFalse(cc.robust_task_pass("keyed_insertion", b))

    def test_episode_score_is_trajectory_wide(self):
        rng = np.random.default_rng(12345)
        ep = ni.generate_episode(rng, max_iter=5, points=32)
        q = cc.episode_nonconformity(ep)
        self.assertTrue(cc.episode_covered(ep, q + 1e-12))

    def test_completion_bounds_monotone_in_q(self):
        rng = np.random.default_rng(7)
        ep = ni.generate_episode(rng, max_iter=2, points=24)
        s = ep.stages[0]
        a = cc.completion_bounds(s, 1.0)
        b = cc.completion_bounds(s, 2.0)
        self.assertTrue(np.all(b >= a))

    def test_invalid_q_fails_closed(self):
        rng = np.random.default_rng(8)
        ep = ni.generate_episode(rng, max_iter=2, points=24)
        b = cc.completion_bounds(ep.stages[0], math.nan)
        self.assertTrue(np.all(np.isinf(b)))
        self.assertFalse(cc.robust_task_pass("top_suction", b))

    def test_infinite_q_produces_hold(self):
        rng = np.random.default_rng(9)
        ep = ni.generate_episode(rng, max_iter=2, points=24)
        k, act, _ = cc.first_completion_stage(ep.stages, "top_suction", math.inf)
        self.assertIsNone(k)
        self.assertFalse(act)


if __name__ == "__main__":
    unittest.main()
