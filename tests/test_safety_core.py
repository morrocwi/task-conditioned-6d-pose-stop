import math
import unittest

from cqts.safety import (
    CertificateNumericsError,
    bonferroni_checkpoint_alpha,
    bonferroni_feasible_max_checkpoints,
    paired_binary_noninferiority,
    safe_exp_error_bound,
    safe_multi_checkpoint_quantiles,
    safe_split_conformal_quantile,
)


class TestSafetyCore(unittest.TestCase):
    def test_review_counterexample_small_n_is_unbounded(self):
        q, rank = safe_split_conformal_quantile([1.0, 2.0], 0.1)
        self.assertEqual(rank, 3)
        self.assertEqual(q, math.inf)

    def test_nan_score_is_rejected(self):
        with self.assertRaises(CertificateNumericsError):
            safe_split_conformal_quantile([1.0, math.nan], 0.1)

    def test_intentional_infinity_bound_stays_infinite(self):
        self.assertEqual(safe_exp_error_bound(0.0, math.inf, 1e-5), math.inf)

    def test_corrupt_q_never_collapses_to_zero(self):
        with self.assertRaises(CertificateNumericsError):
            safe_exp_error_bound(0.0, math.nan, 1e-5)

    def test_one_paired_episode_cannot_certify_five_percent_margin(self):
        out = paired_binary_noninferiority([1], [1], 0.05, 0.95, 1)
        self.assertFalse(out["noninferiority_pass"])

    def test_zero_discordance_has_nonzero_uncertainty(self):
        out = paired_binary_noninferiority([1] * 20, [1] * 20, 0.05, 0.95, 1)
        self.assertLess(out["lower_confidence_bound"], 0.0)

    # --- PROP-CONF-03: Bonferroni-corrected multi-checkpoint conformal band ---

    def test_bonferroni_feasibility_matches_gls_2026_005_diagnosis(self):
        # n=40, alpha=0.1: K'<=4 feasible, K'=5 already infeasible (diagnosis
        # arithmetic, GLS-2026-005/DIAGNOSIS_HYPOTHESIS.md revision).
        self.assertEqual(bonferroni_feasible_max_checkpoints(40, 0.1), 4)

    def test_bonferroni_checkpoint_alpha_splits_evenly(self):
        self.assertAlmostEqual(bonferroni_checkpoint_alpha(0.1, 4), 0.025)

    def test_bonferroni_checkpoint_alpha_rejects_bad_inputs(self):
        with self.assertRaises(CertificateNumericsError):
            bonferroni_checkpoint_alpha(0.1, 0)
        with self.assertRaises(CertificateNumericsError):
            bonferroni_checkpoint_alpha(1.5, 4)

    def test_multi_checkpoint_quantiles_reuse_split_conformal_quantile(self):
        # Same 40 calibration scores at every one of 4 checkpoints, alpha=0.1
        # -> alpha/4=0.025 per checkpoint -> rank=ceil(41*0.975)=40 -> the
        # 40th (largest) score, identically at each checkpoint.
        scores = list(range(1, 41))  # 1..40
        by_checkpoint = {4: scores, 8: scores, 12: scores, 15: scores}
        out = safe_multi_checkpoint_quantiles(by_checkpoint, 0.1)
        self.assertEqual(set(out), {4, 8, 12, 15})
        for k, (q, rank) in out.items():
            self.assertEqual(rank, 40)
            self.assertEqual(q, 40)

    def test_multi_checkpoint_quantiles_fail_closed_when_k_too_large_for_n(self):
        # K'=5 at n=40, alpha=0.1 is provably infeasible (rank=41>40): every
        # checkpoint must come back q=+infinity, not a silently-wrong finite
        # value -- this is the adversarial case the diagnosis explicitly
        # predicted before any code was written.
        scores = list(range(1, 41))
        by_checkpoint = {k: scores for k in (3, 6, 9, 12, 15)}
        out = safe_multi_checkpoint_quantiles(by_checkpoint, 0.1)
        self.assertEqual(len(out), 5)
        for k, (q, rank) in out.items():
            self.assertEqual(q, math.inf)
            self.assertEqual(rank, 41)

    def test_multi_checkpoint_quantiles_matches_manual_per_checkpoint_call(self):
        # The function must not reimplement conformal quantile logic: its
        # output for each checkpoint must be byte-identical to calling
        # safe_split_conformal_quantile directly at alpha/K'.
        scores_a = [float(x) for x in range(10, 50)]
        scores_b = [float(x) * 1.3 for x in range(5, 45)]
        by_checkpoint = {4: scores_a, 15: scores_b}
        out = safe_multi_checkpoint_quantiles(by_checkpoint, 0.1)
        expect_a = safe_split_conformal_quantile(scores_a, 0.05)
        expect_b = safe_split_conformal_quantile(scores_b, 0.05)
        self.assertEqual(out[4], expect_a)
        self.assertEqual(out[15], expect_b)

    def test_multi_checkpoint_quantiles_rejects_empty(self):
        with self.assertRaises(CertificateNumericsError):
            safe_multi_checkpoint_quantiles({}, 0.1)


if __name__ == "__main__":
    unittest.main()
