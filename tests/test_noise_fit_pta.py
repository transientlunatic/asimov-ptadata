"""Real ``enterprise`` PTA-construction tests for ``asimov_ptadata.noise_fit``.

Unlike ``tests/test_noise.py`` (which deliberately keeps the real
enterprise/PINT/PTMCMCSampler fit out of the hermetic pytest suite - see that
module's docstring - and only exercises the Asimov plumbing around it), this
module builds a *real* ``enterprise`` PTA from synthetic PINT TOAs and checks
the actual signal model: per-backend white noise (+ ECORR), red noise, and
DM noise. It stops short of running PTMCMCSampler itself (that's what
``scripts/noise_fit_smoketest.py`` is for) since a single ``get_lnlikelihood``
call is enough to prove the model is well-formed, and running the full
sampler here would make an already fit-heavy test slower for no extra
coverage.

Real network dependency
-------------------------
Building a real ``enterprise.pulsar.Pulsar`` from PINT TOAs resolves a
solar-system ephemeris (DE421) over the network the first time it's needed
(the same dependency ``ptadata reduce`` and the ``scripts/*_smoketest.py``
scripts already carry) - this is unavoidable for a *real* PTA, which is the
whole point of this module (as opposed to ``test_noise.py``'s mocked
plumbing tests). If that resolution fails outright (e.g. a fully airgapped
CI runner), ``setUpModule`` skips the whole module rather than failing it.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import astropy.units as u
import numpy as np
import pint.config
import pint.models
import pint.simulation
import pint.toa

from asimov_ptadata.noise_fit import _build_pta


def setUpModule():
    import astropy.coordinates as coord

    try:
        with coord.solar_system_ephemeris.set("de421"):
            pass
    except Exception as exc:  # pragma: no cover - depends on network access
        raise unittest.SkipTest(f"de421 solar-system ephemeris not resolvable: {exc!r}")


def _make_two_backend_fixture(workdir):
    """
    Build a small synthetic pulsar with TOAs from two distinct fake
    backends (via ``pint.simulation.make_fake_toas_uniform`` + tim-file
    ``-fe``/``-be``/``-f`` flags, merged with ``pint.toa.merge_TOAs``), and
    write its ``.par``/``.tim`` to ``workdir``. Modelled directly on PINT's
    own bundled ``NGC6440E`` example (as every other fixture in this repo's
    tests uses), rather than a from-scratch model, to keep this a real,
    fittable timing model.

    Returns
    -------
    (par_path, tim_path)
    """
    model = pint.models.get_model(pint.config.examplefile("NGC6440E.par"))

    toas_a = pint.simulation.make_fake_toas_uniform(
        55000, 55500, 30, model, freq=1400 * u.MHz, obs="gbt", error=1 * u.us,
        flags={"fe": "430", "be": "ASP", "f": "430_ASP"},
    )
    toas_b = pint.simulation.make_fake_toas_uniform(
        55500, 56000, 30, model, freq=800 * u.MHz, obs="gbt", error=1 * u.us,
        flags={"fe": "L-wide", "be": "PUPPI", "f": "Lwide_PUPPI"},
    )
    toas = pint.toa.merge_TOAs([toas_a, toas_b])

    par_path = workdir / "1748-2021E.par"
    tim_path = workdir / "1748-2021E.tim"
    model.write_parfile(par_path)
    toas.write_TOA_file(tim_path, format="tempo2")
    return par_path, tim_path


class BuildPTATests(unittest.TestCase):
    """Real-PTA tests for ``asimov_ptadata.noise_fit._build_pta``."""

    @classmethod
    def setUpClass(cls):
        cls.workdir = Path(tempfile.mkdtemp(prefix="ptadata-noise-fit-pta-test-"))
        cls.par_path, cls.tim_path = _make_two_backend_fixture(cls.workdir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.workdir, ignore_errors=True)

    def _assert_finite_likelihood(self, pta):
        x0 = np.hstack([p.sample() for p in pta.params]) if pta.params else np.array([])
        # A handful of draws: enterprise's own log-uniform priors have
        # measure-zero unsampled edges, but a single unlucky draw could
        # still occasionally land somewhere with a genuinely non-finite
        # (rather than merely small) likelihood/prior; a few independent
        # draws makes this robust without weakening what's being checked.
        for _ in range(5):
            lnlike = pta.get_lnlikelihood(x0)
            lnprior = pta.get_lnprior(x0)
            self.assertTrue(np.isfinite(lnlike), f"non-finite lnlikelihood: {lnlike}")
            self.assertTrue(np.isfinite(lnprior), f"non-finite lnprior: {lnprior}")
            x0 = np.hstack([p.sample() for p in pta.params]) if pta.params else np.array([])

    def test_default_model_has_per_backend_ecorr_and_dm_noise_params(self):
        psr, pta = _build_pta(self.par_path, [self.tim_path], red_noise_components=5, dm_noise_components=5)

        names = set(pta.param_names)
        prefix = "1748-2021E_"

        for backend in ("430_ASP", "Lwide_PUPPI"):
            for suffix in ("efac", "log10_t2equad", "log10_ecorr"):
                self.assertIn(f"{prefix}{backend}_{suffix}", names)

        self.assertIn(f"{prefix}red_noise_gamma", names)
        self.assertIn(f"{prefix}red_noise_log10_A", names)
        self.assertIn(f"{prefix}dm_gp_gamma", names)
        self.assertIn(f"{prefix}dm_gp_log10_A", names)

        # Timing-model parameters are marginalised, not sampled - they must
        # not show up as free PTA parameters at all.
        self.assertFalse(any("linear_timing_model" in name for name in names))

        self._assert_finite_likelihood(pta)

    def test_use_ecorr_false_omits_ecorr_params(self):
        psr, pta = _build_pta(
            self.par_path, [self.tim_path],
            red_noise_components=5, dm_noise_components=5, use_ecorr=False,
        )
        names = set(pta.param_names)
        self.assertFalse(any("log10_ecorr" in name for name in names))
        self._assert_finite_likelihood(pta)

    def test_use_dm_noise_false_omits_dm_params(self):
        psr, pta = _build_pta(
            self.par_path, [self.tim_path],
            red_noise_components=5, dm_noise_components=5, use_dm_noise=False,
        )
        names = set(pta.param_names)
        self.assertFalse(any("dm_gp" in name for name in names))
        self._assert_finite_likelihood(pta)


if __name__ == "__main__":
    unittest.main()
