import math
import unittest

from cqts.safety import (
    CertificateNumericsError,
    paired_binary_noninferiority,
    safe_exp_error_bound,
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


if __name__ == "__main__":
    unittest.main()
