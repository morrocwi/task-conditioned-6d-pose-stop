import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab.native_persistence import (
    PersistenceConfig,
    persistent_first_act_stage,
    stage_decision_with_variant,
)
from lab.native_sensitivity import NativeConfig

BIG_TASK = {"kind": "box", "mask": [0, 1, 2, 3, 4, 5], "tol": [1e6] * 6}
TINY_TASK = {"kind": "box", "mask": [0, 1, 2, 3, 4, 5], "tol": [1e-9] * 6}

# Small lambda_min on the last axis -> CAP saturates -> perturbation nonzero.
H_LENIENT = np.diag([500.0, 500.0, 500.0, 500.0, 500.0, 0.3])
H_STRICT = np.diag([500.0, 500.0, 500.0, 500.0, 500.0, 0.001])
R0 = np.eye(3)
T0 = np.zeros(3)


def _stage(H):
    return {
        "native": {"hessian": [float(x) for x in H.reshape(-1)], "R": R0.tolist(), "t": T0.tolist()},
        "features": [0.0, -3.0] + [0.0] * 10,  # log-residual = -3 -> rmse ~ exp(-3)
    }


class TestPersistentFirstActStage(unittest.TestCase):
    def setUp(self):
        self.cfg = NativeConfig(lambda_ref=0.34, cap=0.005)

    def test_theta_one_matches_single_instant_check(self):
        stages = [_stage(H_LENIENT)]
        pcfg = PersistenceConfig(theta=1)
        idx, act, audit = persistent_first_act_stage(stages, BIG_TASK, self.cfg, pcfg)
        self.assertEqual(idx, 0)
        self.assertTrue(act)

    def test_theta_gates_until_streak_reached(self):
        stages = [_stage(H_LENIENT) for _ in range(5)]
        pcfg = PersistenceConfig(theta=3)
        idx, act, audit = persistent_first_act_stage(stages, BIG_TASK, self.cfg, pcfg)
        self.assertEqual(idx, 2)  # 0-indexed: 3rd consecutive pass
        self.assertTrue(act)
        self.assertEqual(len(audit), 3)
        self.assertEqual([a["streak_M_k"] for a in audit], [1, 2, 3])

    def test_streak_resets_on_a_failing_stage(self):
        # Construct a genuine pass/fail/pass pattern with a FIXED task by varying H's smallest
        # eigenvalue (which controls the saturation of m=min(c_scale/lambda_min, cap)):
        #   lambda_min=2.0 -> m=min(0.01/2.0, 0.01)=0.005  <= tol=0.006 -> PASS
        #   lambda_min=0.5 -> m=min(0.01/0.5, 0.01)=0.01    >  tol=0.006 -> FAIL
        cfg2 = NativeConfig(lambda_ref=1.0, cap=0.01)
        mid_task = {"kind": "box", "mask": [0, 1, 2, 3, 4, 5], "tol": [1e6, 1e6, 0.006, 1e6, 1e6, 1e6]}
        H_pass = np.diag([500.0, 500.0, 500.0, 500.0, 500.0, 2.0])
        H_fail = np.diag([500.0, 500.0, 500.0, 500.0, 500.0, 0.5])
        stages = [_stage(H_pass), _stage(H_pass), _stage(H_fail), _stage(H_pass), _stage(H_pass)]
        pcfg = PersistenceConfig(theta=2)
        idx, act, audit = persistent_first_act_stage(stages, mid_task, cfg2, pcfg)
        # theta=2 is reached already at the second stage (streak [1,2]); the function returns as
        # soon as the threshold is met, so the audit trail is truncated to those 2 stages -- the
        # later fail-stage is never even evaluated for this episode.
        self.assertEqual([a["act"] for a in audit], [True, True])
        self.assertEqual([a["streak_M_k"] for a in audit], [1, 2])
        self.assertEqual(idx, 1)
        self.assertTrue(act)

    def test_streak_resets_and_theta_reached_only_after_second_streak(self):
        cfg2 = NativeConfig(lambda_ref=1.0, cap=0.01)
        mid_task = {"kind": "box", "mask": [0, 1, 2, 3, 4, 5], "tol": [1e6, 1e6, 0.006, 1e6, 1e6, 1e6]}
        H_pass = np.diag([500.0, 500.0, 500.0, 500.0, 500.0, 2.0])
        H_fail = np.diag([500.0, 500.0, 500.0, 500.0, 500.0, 0.5])
        # single pass, then fail (resets), then needs 2 more passes to reach theta=2
        stages = [_stage(H_pass), _stage(H_fail), _stage(H_pass), _stage(H_pass)]
        pcfg = PersistenceConfig(theta=2)
        idx, act, audit = persistent_first_act_stage(stages, mid_task, cfg2, pcfg)
        self.assertEqual([a["streak_M_k"] for a in audit], [1, 0, 1, 2])
        self.assertEqual(idx, 3)
        self.assertTrue(act)

    def test_never_reaches_theta_returns_none(self):
        stages = [_stage(H_STRICT) for _ in range(4)]
        pcfg = PersistenceConfig(theta=2)
        idx, act, audit = persistent_first_act_stage(stages, TINY_TASK, self.cfg, pcfg)
        self.assertIsNone(idx)
        self.assertFalse(act)
        self.assertEqual(len(audit), 4)

    def test_streak_counter_resets_to_zero_after_failure(self):
        # BIG_TASK always passes (huge tolerance); construct a synthetic streak by hand to check
        # the resetting-streak arithmetic itself via stage_decision_with_variant + manual loop,
        # since BIG/TINY tasks alone can't easily produce a pass-fail-pass pattern with this
        # backend's real perturbation geometry.
        pcfg = PersistenceConfig(theta=10**9)  # unreachable, forces full audit trail
        stages = [_stage(H_LENIENT) for _ in range(4)]
        idx, act, audit = persistent_first_act_stage(stages, BIG_TASK, self.cfg, pcfg)
        self.assertIsNone(idx)
        self.assertEqual([a["streak_M_k"] for a in audit], [1, 2, 3, 4])


class TestVariantB(unittest.TestCase):
    def setUp(self):
        self.cfg = NativeConfig(lambda_ref=0.34, cap=0.005)

    def test_variant_a_identical_to_native_sensitivity_stage_decision(self):
        from lab.native_sensitivity import stage_decision as base_decision

        pcfg = PersistenceConfig(theta=1)
        d_variant = stage_decision_with_variant(R0, T0, H_LENIENT, BIG_TASK, self.cfg, pcfg, features=None)
        d_base = base_decision(R0, T0, H_LENIENT, BIG_TASK, self.cfg)
        self.assertEqual(d_variant["act"], d_base["act"])
        self.assertAlmostEqual(d_variant["magnitude"], d_base["magnitude"], places=12)

    def test_variant_b_scales_magnitude_up_when_residual_exceeds_reference(self):
        pcfg = PersistenceConfig(theta=1, residual_scale_ref=0.01, residual_cap_multiplier=2.0)
        # features[1] = log(residual+eps); choose residual = 0.05 (5x the reference 0.01)
        features_high_residual = [0.0, float(np.log(0.05 + 1e-9))] + [0.0] * 10
        d = stage_decision_with_variant(R0, T0, H_LENIENT, BIG_TASK, self.cfg, pcfg, features=features_high_residual)
        self.assertGreater(d["magnitude"], d["magnitude_base"])
        self.assertLessEqual(d["magnitude"], pcfg.residual_cap_multiplier * self.cfg.cap + 1e-15)
        self.assertAlmostEqual(d["residual_scale_applied"], 5.0, places=3)

    def test_variant_b_never_scales_down_below_base_magnitude(self):
        pcfg = PersistenceConfig(theta=1, residual_scale_ref=0.05, residual_cap_multiplier=2.0)
        # residual well below reference -> scale factor floors at 1.0, no shrinkage
        features_low_residual = [0.0, float(np.log(0.001 + 1e-9))] + [0.0] * 10
        d = stage_decision_with_variant(R0, T0, H_LENIENT, BIG_TASK, self.cfg, pcfg, features=features_low_residual)
        self.assertAlmostEqual(d["magnitude"], d["magnitude_base"], places=12)
        self.assertAlmostEqual(d["residual_scale_applied"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
