"""Fast, hermetic unit tests for the sampling helpers in
``asimov_ptadata.noise_fit``: the convergence diagnostics
(``effective_sample_size``, ``split_shift``, ``convergence_summary``), the
PTMCMC jump groups (``_param_groups``), and the optimised starting point
(``_initial_point``).

Unlike ``tests/test_noise_fit_pta.py`` (a real ``enterprise`` PTA from
synthetic PINT TOAs) and ``tests/test_noise.py`` (the Asimov plumbing around
a mocked fit), nothing here builds a PTA, calls ``enterprise``, or runs
PTMCMCSampler - these are pure-numpy/pure-Python functions, exercised
directly with small seeded arrays and a tiny fake PTA-like object, so the
whole module runs in well under a second.
"""

import unittest
from types import SimpleNamespace

import numpy as np

from asimov_ptadata.noise_fit import (
    _initial_point,
    _param_groups,
    convergence_summary,
    effective_sample_size,
    split_shift,
)


class EffectiveSampleSizeTests(unittest.TestCase):
    """Tests for ``effective_sample_size``."""

    def test_iid_chain_has_ess_close_to_n(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=2000)
        ess = effective_sample_size(x)
        self.assertGreater(ess, 0.5 * len(x))
        self.assertLessEqual(ess, len(x) * 1.5)  # sanity: not absurdly inflated

    def test_autocorrelated_chain_has_much_smaller_ess(self):
        rng = np.random.default_rng(1)
        n = 5000
        phi = 0.99
        innovations = rng.normal(size=n) * np.sqrt(1 - phi**2)
        x = np.empty(n)
        x[0] = innovations[0]
        for i in range(1, n):
            x[i] = phi * x[i - 1] + innovations[i]

        ess_ar1 = effective_sample_size(x)

        rng2 = np.random.default_rng(2)
        ess_iid = effective_sample_size(rng2.normal(size=n))

        self.assertLess(ess_ar1, 0.1 * ess_iid)

    def test_constant_chain_returns_n(self):
        # 3.14 isn't exactly representable, so np.var of a flat chain of it is
        # ~1e-30 rather than 0; the flat-chain check must still catch it.
        for value in (2.5, 3.14):
            self.assertEqual(effective_sample_size(np.full(50, value)), 50.0)
            self.assertEqual(split_shift(np.full(50, value)), 0.0)


class SplitShiftTests(unittest.TestCase):
    """Tests for ``split_shift``."""

    def test_stationary_chain_has_small_shift(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=2000)
        self.assertLess(split_shift(x), 0.2)

    def test_second_half_offset_gives_large_shift(self):
        rng = np.random.default_rng(4)
        x = np.concatenate([rng.normal(size=1000), rng.normal(loc=5.0, size=1000)])
        self.assertGreater(split_shift(x), 1.0)


class ConvergenceSummaryTests(unittest.TestCase):
    """Tests for ``convergence_summary``."""

    def test_reports_thresholds_and_per_parameter_diagnostics(self):
        rng = np.random.default_rng(5)
        chain = rng.normal(size=(3000, 2))
        names = ["a", "b"]

        info, converged = convergence_summary(chain, names, min_ess=200.0, max_split_shift=0.3)

        self.assertEqual(info["min_ess"], 200.0)
        self.assertEqual(info["max_split_shift"], 0.3)
        self.assertEqual(set(info["parameters"]), set(names))
        for name in names:
            self.assertIn("ess", info["parameters"][name])
            self.assertIn("split_shift", info["parameters"][name])

    def test_converged_true_for_well_mixed_iid_chains(self):
        rng = np.random.default_rng(6)
        chain = rng.normal(size=(3000, 2))
        _, converged = convergence_summary(chain, ["a", "b"], min_ess=200.0, max_split_shift=0.3)
        self.assertTrue(converged)

    def test_converged_false_when_one_parameter_fails_split_shift(self):
        rng = np.random.default_rng(7)
        good = rng.normal(size=3000)
        bad = np.concatenate([np.zeros(1500), np.full(1500, 5.0)])
        chain = np.column_stack([good, bad])

        info, converged = convergence_summary(chain, ["good", "bad"], min_ess=200.0, max_split_shift=0.3)

        self.assertFalse(converged)
        self.assertLessEqual(info["parameters"]["good"]["split_shift"], 0.3)
        self.assertGreater(info["parameters"]["bad"]["split_shift"], 0.3)

    def test_converged_false_when_one_parameter_fails_ess(self):
        rng = np.random.default_rng(8)
        good = rng.normal(size=3000)
        phi = 0.995
        innovations = rng.normal(size=3000) * np.sqrt(1 - phi**2)
        bad = np.empty(3000)
        bad[0] = innovations[0]
        for i in range(1, 3000):
            bad[i] = phi * bad[i - 1] + innovations[i]
        chain = np.column_stack([good, bad])

        info, converged = convergence_summary(chain, ["good", "bad"], min_ess=200.0, max_split_shift=0.3)

        self.assertFalse(converged)
        self.assertGreaterEqual(info["parameters"]["good"]["ess"], 200.0)
        self.assertLess(info["parameters"]["bad"]["ess"], 200.0)


class ParamGroupsTests(unittest.TestCase):
    """Tests for ``_param_groups``."""

    def test_groups_cover_all_parameters_backends_and_red_dm(self):
        names = [
            "J0000+0000_A_efac",
            "J0000+0000_A_log10_t2equad",
            "J0000+0000_A_log10_ecorr",
            "J0000+0000_B_efac",
            "J0000+0000_B_log10_t2equad",
            "J0000+0000_dm_gp_gamma",
            "J0000+0000_dm_gp_log10_A",
            "J0000+0000_red_noise_gamma",
            "J0000+0000_red_noise_log10_A",
        ]

        groups = _param_groups(names)
        group_sets = [set(g) for g in groups]

        self.assertIn(set(range(len(names))), group_sets)  # every parameter
        self.assertIn({0, 1, 2}, group_sets)  # backend A's 3 white-noise indices
        self.assertIn({3, 4}, group_sets)  # backend B's 2 white-noise indices
        self.assertIn({7, 8}, group_sets)  # the red-noise pair
        self.assertIn({5, 6}, group_sets)  # the DM-noise pair
        self.assertIn({5, 6, 7, 8}, group_sets)  # red + DM together
        self.assertIn({0, 1, 2, 3, 4}, group_sets)  # every white-noise index


class InitialPointTests(unittest.TestCase):
    """Tests for ``_initial_point``, against a tiny fake PTA-like object."""

    class _FakeParam:
        def __init__(self, pmin, pmax):
            self.prior = SimpleNamespace(_defaults={"pmin": pmin, "pmax": pmax})

    class _FakePTA:
        """Enough of enterprise's PTA interface for ``_initial_point``: a
        smooth, separable log-likelihood peaked at known red/DM values, and a
        box-prior ``get_lnprior``."""

        def __init__(self, names, bounds, target):
            self.param_names = names
            self.params = [InitialPointTests._FakeParam(*bounds[n]) for n in names]
            self._bounds = bounds
            self._target = target

        def get_lnprior(self, x):
            for xi, name in zip(x, self.param_names):
                lo, hi = self._bounds[name]
                if not (lo <= xi <= hi):
                    return -np.inf
            return 0.0

        def get_lnlikelihood(self, x):
            total = 0.0
            for xi, name in zip(x, self.param_names):
                if name in self._target:
                    total -= (xi - self._target[name]) ** 2
            return total

    def setUp(self):
        self.names = [
            "PSR_A_efac",
            "PSR_A_log10_t2equad",
            "PSR_red_noise_log10_A",
            "PSR_red_noise_gamma",
            "PSR_dm_gp_log10_A",
            "PSR_dm_gp_gamma",
        ]
        self.bounds = {
            "PSR_A_efac": (0.1, 5.0),
            "PSR_A_log10_t2equad": (-10.0, -5.0),
            "PSR_red_noise_log10_A": (-20.0, -11.0),
            "PSR_red_noise_gamma": (0.0, 7.0),
            "PSR_dm_gp_log10_A": (-20.0, -11.0),
            "PSR_dm_gp_gamma": (0.0, 7.0),
        }
        self.target = {
            "PSR_red_noise_log10_A": -14.2,
            "PSR_red_noise_gamma": 3.5,
            "PSR_dm_gp_log10_A": -13.0,
            "PSR_dm_gp_gamma": 2.0,
        }
        self.pta = self._FakePTA(self.names, self.bounds, self.target)

    def test_white_noise_is_optimised_per_backend_too(self):
        # Leaving white noise at its prior floor let a flat "red" process
        # stand in for it (J1713+0747), so it must be optimised as well.
        target = dict(self.target, PSR_A_efac=2.3, **{"PSR_A_log10_t2equad": -6.5})
        pta = self._FakePTA(self.names, self.bounds, target)
        x0 = _initial_point(pta, np.array([0.5, -9.0, -15.0, 4.0, -16.0, 5.0]))
        for name, value in target.items():
            self.assertAlmostEqual(x0[self.names.index(name)], value, delta=0.1, msg=name)

    def test_red_and_dm_parameters_land_near_the_likelihood_peak(self):
        x_prior = np.array([2.0, -7.0, -15.0, 4.0, -16.0, 5.0])
        x0 = _initial_point(self.pta, x_prior)

        for name, target_value in self.target.items():
            idx = self.names.index(name)
            self.assertAlmostEqual(x0[idx], target_value, delta=0.1)


if __name__ == "__main__":
    unittest.main()
