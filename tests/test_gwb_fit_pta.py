"""Real ``enterprise`` PTA-construction tests for ``asimov_ptadata.gwb_fit``.

Mirrors ``tests/test_noise_fit_pta.py``'s split from the Asimov-plumbing
tests in ``tests/test_gwb.py``: this builds a real, joint ``enterprise`` PTA
from synthetic PINT TOAs via ``gwb_fit._build_joint_pta`` and checks the
actual fixed-noise + common-process signal model, rather than mocking the
scheduler and asimov ledger machinery around it. See that module's docstring
for the shared real-network-ephemeris caveat (``setUpModule`` skips this
module if DE421 isn't resolvable at all).
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from asimov_ptadata.gwb_fit import FIXED_NOISE_PARAMS, _build_joint_pta
from tests.test_noise_fit_pta import _make_two_backend_fixture


def setUpModule():
    import astropy.coordinates as coord

    try:
        with coord.solar_system_ephemeris.set("de421"):
            pass
    except Exception as exc:  # pragma: no cover - depends on network access
        raise unittest.SkipTest(f"de421 solar-system ephemeris not resolvable: {exc!r}")


class BuildJointPTATests(unittest.TestCase):
    """Real-PTA tests for ``asimov_ptadata.gwb_fit._build_joint_pta``."""

    @classmethod
    def setUpClass(cls):
        cls.workdir = Path(tempfile.mkdtemp(prefix="ptadata-gwb-fit-pta-test-"))
        cls.par_path, cls.tim_path = _make_two_backend_fixture(cls.workdir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.workdir, ignore_errors=True)

    def _assert_finite_likelihood(self, pta):
        x0 = np.hstack([p.sample() for p in pta.params]) if pta.params else np.array([])
        for _ in range(5):
            lnlike = pta.get_lnlikelihood(x0)
            lnprior = pta.get_lnprior(x0)
            self.assertTrue(np.isfinite(lnlike), f"non-finite lnlikelihood: {lnlike}")
            self.assertTrue(np.isfinite(lnprior), f"non-finite lnprior: {lnprior}")
            x0 = np.hstack([p.sample() for p in pta.params]) if pta.params else np.array([])

    def test_fixed_noise_leaves_only_the_shared_common_process_free(self):
        pulsars = [{
            "name": "1748-2021E",
            "par": self.par_path,
            "tim": [self.tim_path],
            "noise_params": {
                "430_ASP_efac": 1.1,
                "430_ASP_log10_t2equad": -7.0,
                "430_ASP_log10_ecorr": -7.2,
                "Lwide_PUPPI_efac": 0.95,
                "Lwide_PUPPI_log10_t2equad": -7.3,
                "Lwide_PUPPI_log10_ecorr": -7.6,
                "red_noise_gamma": 3.5,
                "red_noise_log10_A": -14.0,
                "dm_gp_gamma": 2.1,
                "dm_gp_log10_A": -13.5,
            },
        }]

        psrs, pta = _build_joint_pta(pulsars, red_noise_components=5, dm_noise_components=5, gwb_components=5)

        # Every per-pulsar noise parameter is fixed - only the shared
        # common-process pair is left free to sample.
        self.assertEqual(sorted(pta.param_names), ["gwb_gamma", "gwb_log10_A"])
        self._assert_finite_likelihood(pta)

    def test_legacy_fixed_noise_dict_still_builds_a_working_pta(self):
        # The old, pre-per-backend FIXED_NOISE_PARAMS shape (no backend
        # suffix, no ECORR, no DM noise) - confirms the backwards
        # compatibility path documented in gwb_fit.py.
        self.assertEqual(set(FIXED_NOISE_PARAMS), {"efac", "log10_t2equad", "red_noise_gamma", "red_noise_log10_A"})
        pulsars = [{
            "name": "1748-2021E",
            "par": self.par_path,
            "tim": [self.tim_path],
            "noise_params": {
                "efac": 1.1,
                "log10_t2equad": -7.0,
                "red_noise_gamma": 3.5,
                "red_noise_log10_A": -14.0,
            },
        }]

        psrs, pta = _build_joint_pta(pulsars, red_noise_components=5, dm_noise_components=5, gwb_components=5)

        self.assertEqual(sorted(pta.param_names), ["gwb_gamma", "gwb_log10_A"])
        self._assert_finite_likelihood(pta)


if __name__ == "__main__":
    unittest.main()
