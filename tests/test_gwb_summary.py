"""The GWB report's posterior summaries and amplitude-constrained verdict
(``asimov_ptadata.gwb_fit``), on synthetic chains."""

import unittest

import numpy as np

from asimov_ptadata.gwb_fit import GWB_LOG10_A_PRIOR, _amplitude_constrained, _summaries


class GWBSummaryTests(unittest.TestCase):
    def test_summaries_give_percentiles_and_sd(self):
        chain = np.random.default_rng(0).normal(-14.6, 0.2, size=(5000, 1))
        s = _summaries(chain, ["gwb_log10_A"])["gwb_log10_A"]
        self.assertAlmostEqual(s["p50"], -14.6, delta=0.02)
        self.assertAlmostEqual(s["sd"], 0.2, delta=0.02)
        self.assertLess(s["p05"], s["p50"])
        self.assertLess(s["p50"], s["p95"])

    def test_a_narrow_posterior_is_constrained(self):
        chain = np.random.default_rng(1).normal(-14.6, 0.3, size=(5000, 1))
        self.assertTrue(_amplitude_constrained(chain, ["gwb_log10_A"]))

    def test_a_prior_like_posterior_is_not_constrained(self):
        # The 5-pulsar IPTA DR2 pilot: -17.4 +/- 1.6 on U(-20, -11).
        lo, hi = GWB_LOG10_A_PRIOR
        chain = np.clip(np.random.default_rng(2).normal(-17.4, 1.6, size=(5000, 1)), lo, hi)
        self.assertFalse(_amplitude_constrained(chain, ["gwb_log10_A"]))

    def test_no_amplitude_no_verdict(self):
        self.assertIsNone(_amplitude_constrained(np.zeros((10, 1)), ["gwb_gamma"]))


if __name__ == "__main__":
    unittest.main()
