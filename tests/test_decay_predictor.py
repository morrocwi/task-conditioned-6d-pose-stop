import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab.bop_icp_backend import ICPStage, normal_equations_H, run_icp
from lab.decay_predictor import (
    DecayModel,
    correspondence_graph_n_and_diameter,
    decay_predicted_log_error,
    kappa_rho_from_H,
    mohar_floor_4_over_nD,
)


class TestKappaRho(unittest.TestCase):
    def test_well_conditioned_identity_scaled_H(self):
        H = np.diag([1.0, 1.0, 1.0, 4.0, 4.0, 4.0])
        kr = kappa_rho_from_H(H)
        self.assertFalse(kr.degenerate)
        self.assertAlmostEqual(kr.kappa, 4.0, places=6)
        self.assertAlmostEqual(kr.rho, 3.0 / 5.0, places=6)
        self.assertTrue(0.0 <= kr.rho <= 1.0)

    def test_singular_H_is_fail_closed(self):
        H = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 0.0])  # one pose direction unobserved
        kr = kappa_rho_from_H(H)
        self.assertTrue(kr.degenerate)
        self.assertEqual(kr.kappa, math.inf)
        self.assertEqual(kr.rho, 1.0)  # no contraction guaranteed, never a wrong finite number

    def test_symmetrizes_float_roundoff(self):
        H = np.eye(6) + 1e-14 * (np.arange(36).reshape(6, 6))
        kr = kappa_rho_from_H(H)
        self.assertTrue(math.isfinite(kr.kappa))

    def test_rho_never_exceeds_unity_or_goes_negative(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            A = rng.normal(size=(6, 6))
            H = A.T @ A + 1e-6 * np.eye(6)
            kr = kappa_rho_from_H(H)
            self.assertGreaterEqual(kr.rho, 0.0)
            self.assertLessEqual(kr.rho, 1.0)


class TestDecayPredictedLogError(unittest.TestCase):
    def _stage(self, k, K, rho, residual, degenerate=False):
        return {
            "k": k,
            "decay": {
                "k": k, "K": K, "rho": rho, "kappa": None if degenerate else 1.0 / max(1e-9, 1 - rho),
                "degenerate": degenerate, "lambda_min": 0.0, "lambda_max": 0.0,
                "residual_axes": residual,
            },
        }

    def test_final_stage_reduces_to_floored_residual(self):
        model = DecayModel(error_floor=np.array([1e-5] * 3 + [1e-4] * 3))
        stage = self._stage(k=10, K=10, rho=0.9, residual=[0.02, 0.0, 0.0, 0.0, 0.0, 0.0])
        value = decay_predicted_log_error(stage, axis=0, model=model)
        self.assertAlmostEqual(value, math.log(0.02), places=9)

    def test_decays_geometrically_with_steps_remaining(self):
        model = DecayModel(error_floor=np.array([1e-5] * 6))
        residual = [0.02] * 6
        early = decay_predicted_log_error(self._stage(0, 10, 0.9, residual), 0, model)
        late = decay_predicted_log_error(self._stage(9, 10, 0.9, residual), 0, model)
        self.assertLess(early, late)  # more steps remaining -> smaller predicted error (rho<1)

    def test_missing_decay_field_raises(self):
        model = DecayModel(error_floor=np.array([1e-5] * 6))
        with self.assertRaises(ValueError):
            decay_predicted_log_error({"k": 0}, 0, model)

    def test_floor_applies_when_residual_below_it(self):
        model = DecayModel(error_floor=np.array([1e-3] * 6))
        stage = self._stage(k=5, K=5, rho=0.9, residual=[1e-9] * 6)
        value = decay_predicted_log_error(stage, axis=0, model=model)
        self.assertAlmostEqual(value, math.log(1e-3), places=9)


class TestCorrespondenceGraphDiagnostic(unittest.TestCase):
    def test_path_like_points_have_expected_diameter_order(self):
        # 10 collinear points -> a k=1-nn chain graph has diameter close to n-1
        pts = np.stack([np.arange(10, dtype=float), np.zeros(10), np.zeros(10)], axis=1)
        g = correspondence_graph_n_and_diameter(pts, k_nn=1)
        self.assertEqual(g["n"], 10)
        self.assertTrue(g["connected"])
        self.assertGreaterEqual(g["diameter"], 9)

    def test_too_few_points_reports_undefined_diameter(self):
        g = correspondence_graph_n_and_diameter(np.zeros((1, 3)))
        self.assertIsNone(g["diameter"])
        self.assertFalse(g["connected"])

    def test_mohar_floor_none_when_disconnected_or_degenerate(self):
        self.assertIsNone(mohar_floor_4_over_nD(5, None))
        self.assertIsNone(mohar_floor_4_over_nD(0, 3))
        self.assertAlmostEqual(mohar_floor_4_over_nD(4, D=16), 4.0 / (4 * 16))


class TestRealICPExposesFixedSizeHessian(unittest.TestCase):
    """H_k~graph-Laplacian structural finding, exercised directly (not just
    argued): H stays 6x6 regardless of correspondence count n."""

    def test_H_is_always_6x6_regardless_of_n(self):
        rng = np.random.default_rng(1)
        model_small = rng.normal(size=(15, 3)) * 0.05
        model_big = rng.normal(size=(150, 3)) * 0.05
        R, t = np.eye(3), np.zeros(3)
        H_small = normal_equations_H(model_small, R, t, model_small)
        H_big = normal_equations_H(model_big, R, t, model_big)
        self.assertEqual(H_small.shape, (6, 6))
        self.assertEqual(H_big.shape, (6, 6))

    def test_run_icp_populates_hessian_field(self):
        rng = np.random.default_rng(2)
        model = rng.normal(size=(40, 3)) * 0.05
        scene = model.copy()
        R0, t0 = np.eye(3), np.array([0.01, 0.0, 0.0])
        stages = run_icp(model, scene, R0, t0, max_iter=3)
        self.assertTrue(all(isinstance(s, ICPStage) for s in stages))
        self.assertTrue(all(s.hessian is not None for s in stages))
        self.assertTrue(all(s.hessian.shape == (6, 6) for s in stages))


if __name__ == "__main__":
    unittest.main()
