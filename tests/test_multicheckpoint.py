import importlib.util
import inspect
import math
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("labrun", ROOT / "lab" / "run_real_system.py")
labrun = importlib.util.module_from_spec(spec)
spec.loader.exec_module(labrun)

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from lab import multicheckpoint as mc  # noqa: E402


def _stage(k, features, err, incremental_ms=1.0, estimator_stop=False):
    return (
        {"k": k, "features": features, "incremental_ms": incremental_ms, "estimator_stop": estimator_stop},
        err,
    )


def _episode(eid, n_stages=16, feat_dim=2, seed=0):
    rng = np.random.default_rng(seed)
    stages, errors = [], []
    for k in range(n_stages):
        feat = list(rng.normal(size=feat_dim) + [0.0, k / n_stages][:feat_dim])
        err = list(np.abs(rng.normal(scale=0.01, size=6)))
        stages.append({"k": k, "features": feat, "incremental_ms": 1.0, "estimator_stop": (k == n_stages - 1)})
        errors.append(err)
    return {"episode_id": eid, "stages": stages, "oracle": {"abs_pose_error_6d": errors}}


def _make_episodes(n, feat_dim=2, seed0=0):
    return [_episode(f"ep{i}", feat_dim=feat_dim, seed=seed0 + i) for i in range(n)]


class TestMultiCheckpoint(unittest.TestCase):
    def test_gate_has_no_oracle_argument(self):
        params = set(inspect.signature(mc.first_certificate_checkpoint).parameters)
        self.assertEqual(params, {"ep", "spec", "model", "checkpoints", "q_by_checkpoint"})

    def test_validate_checkpoints_rejects_unsorted(self):
        stages = [{"k": k} for k in range(16)]
        with self.assertRaises(ValueError):
            mc.validate_checkpoints([8, 4, 12, 15], stages)

    def test_validate_checkpoints_rejects_missing_k(self):
        stages = [{"k": k} for k in range(10)]
        with self.assertRaises(ValueError):
            mc.validate_checkpoints([4, 8, 12, 15], stages)

    def test_validate_checkpoints_accepts_valid(self):
        stages = [{"k": k} for k in range(16)]
        self.assertEqual(mc.validate_checkpoints([4, 8, 12, 15], stages), [4, 8, 12, 15])

    def test_nonconformity_at_checkpoint_uses_single_stage_coordinate_max_only(self):
        train = _make_episodes(20, feat_dim=2)
        model = labrun.fit_model(train, [1e-5] * 3 + [1e-4] * 3)
        ep = train[0]
        # score at checkpoint k should differ, in general, from the
        # whole-trajectory (stage-max) score, because it looks at ONE stage.
        whole = labrun.nonconformity(ep, model)
        at4 = mc.nonconformity_at_checkpoint(ep, model, 4)
        self.assertTrue(math.isfinite(at4))
        self.assertLessEqual(at4, whole + 1e-9)  # single-stage max <= whole-trajectory max

    def test_stage_and_error_at_k_fails_closed_on_missing_checkpoint(self):
        ep = _episode("x", n_stages=5)
        with self.assertRaises(ValueError):
            mc.stage_and_error_at_k(ep, 15)

    def test_calibrate_checkpoints_reuses_safe_split_conformal_quantile(self):
        train = _make_episodes(40, feat_dim=2, seed0=100)
        cal = _make_episodes(40, feat_dim=2, seed0=200)
        model = labrun.fit_model(train, [1e-5] * 3 + [1e-4] * 3)
        checkpoints = [4, 8, 12, 15]
        out = mc.calibrate_checkpoints(cal, model, checkpoints, 0.1)
        self.assertEqual(set(out), set(checkpoints))
        for k, (q, rank) in out.items():
            self.assertTrue(math.isfinite(q) or q == math.inf)
            self.assertGreaterEqual(rank, 1)

    def test_feasible_k4_at_n40_is_never_forced_to_infinity_by_construction(self):
        # This is a structural check: K'=4 at n=40, alpha=0.1 gives per-checkpoint
        # alpha=0.025 -> rank=ceil(41*0.975)=40<=40 -> feasible, matching
        # cqts.safety.bonferroni_feasible_max_checkpoints(40, 0.1) == 4.
        from cqts.safety import bonferroni_feasible_max_checkpoints

        self.assertEqual(bonferroni_feasible_max_checkpoints(40, 0.1), 4)

    def test_adversarial_k5_at_n40_fails_closed_to_infinity_every_checkpoint(self):
        # The diagnosis proved K'=5 is already infeasible at n=40, alpha=0.1
        # (rank=41>40). Verify the new code path correctly returns q=+infinity
        # at EVERY checkpoint in this case, rather than silently computing
        # something else -- this is the exact adversarial case requested.
        train = _make_episodes(40, feat_dim=2, seed0=300)
        cal = _make_episodes(40, feat_dim=2, seed0=400)
        model = labrun.fit_model(train, [1e-5] * 3 + [1e-4] * 3)
        checkpoints = [1, 4, 8, 12, 15]  # K'=5
        out = mc.calibrate_checkpoints(cal, model, checkpoints, 0.1)
        self.assertEqual(len(out), 5)
        for k, (q, rank) in out.items():
            self.assertEqual(q, math.inf, f"checkpoint {k} should be forced to +infinity when K'=5 > feasible max 4")
            self.assertEqual(rank, 41)

    def test_adversarial_k5_never_certifies_and_stays_full_hold(self):
        # End-to-end: with K'=5 forced-infinite calibration, no episode should
        # ever certify at any checkpoint -- must fail closed to HOLD, not
        # silently pass with an unbounded envelope.
        train = _make_episodes(40, feat_dim=2, seed0=500)
        cal = _make_episodes(40, feat_dim=2, seed0=600)
        test = _make_episodes(20, feat_dim=2, seed0=700)
        model = labrun.fit_model(train, [1e-5] * 3 + [1e-4] * 3)
        checkpoints = [1, 4, 8, 12, 15]
        q_by_checkpoint = mc.calibrate_checkpoints(cal, model, checkpoints, 0.1)
        spec = {"kind": "box", "mask": [0, 1, 2, 3, 4, 5], "tol": [1e9] * 6}
        for ep in test:
            idx, act = mc.first_certificate_checkpoint(ep, spec, model, checkpoints, q_by_checkpoint)
            # Even a trivially-permissive task spec cannot certify when every
            # checkpoint's q is +infinity (completion_bounds returns +inf,
            # task_pass with tol=1e9 and bound=inf is False, since inf>1e9).
            self.assertFalse(act)
            self.assertIsNone(idx)

    def test_evaluate_multicheckpoint_reports_all_checkpoints_covered_rate(self):
        train = _make_episodes(40, feat_dim=2, seed0=800)
        cal = _make_episodes(40, feat_dim=2, seed0=900)
        test = _make_episodes(20, feat_dim=2, seed0=1000)
        model = labrun.fit_model(train, [1e-5] * 3 + [1e-4] * 3)
        checkpoints = [4, 8, 12, 15]
        q_by_checkpoint = mc.calibrate_checkpoints(cal, model, checkpoints, 0.1)
        tasks = {"loose": {"kind": "box", "mask": [0, 1, 2, 3, 4, 5], "tol": [1e9] * 6}}
        inference = {"margin": 0.05, "confidence_level": 0.95, "minimum_test_episodes": 1}
        ev = mc.evaluate_multicheckpoint(test, tasks, model, checkpoints, q_by_checkpoint, inference, 200, "trajectory_prefix_estimate")
        self.assertIn("all_checkpoints_covered_rate", ev)
        self.assertEqual(set(ev["per_checkpoint"]), {4, 8, 12, 15})
        self.assertIn("loose", ev["tasks"])


if __name__ == "__main__":
    unittest.main()
