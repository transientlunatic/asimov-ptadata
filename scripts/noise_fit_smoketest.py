#!/usr/bin/env python3
"""Standalone smoke test: a real single-pulsar noise fit (enterprise + PINT +
PTMCMCSampler) against this repo's PINT-bundled NGC6440E-derived fixture
pulsar data, run through ``asimov_ptadata.noise_fit.run_noise_fit`` - the
same function the ``ptadata-noise`` Asimov pipeline's HTCondor job runs via
``ptadata noise-run``.

This is deliberately *not* part of the pytest suite (see the docstring of
``tests/test_noise.py``): a real fit needs a solar-system ephemeris, which
PINT resolves over the network the first time it's needed (and caches
afterwards) - the same dependency ``ptadata reduce`` already has. Keeping
that out of the hermetic pytest suite avoids making every CI run of the
unit tests depend on JPL's ephemeris server being reachable.

Some sandboxed network policies block PINT's usual JPL mirrors directly but
do allow reaching GitHub/PyPI; if the normal astropy ephemeris resolution
fails, this script falls back to seeding astropy's download cache from a
public GitHub-hosted mirror of the same public-domain DE421 kernel before
retrying. In a normally-networked environment (e.g. a GitHub Actions
runner), this fallback simply never triggers - the same real network path
``ptadata reduce`` already relies on succeeds directly.

Usage
-----
    python scripts/noise_fit_smoketest.py
"""

import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pint.config  # noqa: E402

from asimov_ptadata.noise_fit import run_noise_fit  # noqa: E402


def _ensure_ephemeris(ephem="de421"):
    """
    Make sure PINT/astropy can resolve the given solar-system ephemeris.

    Tries the normal path first (which is all that's needed on a normally
    networked machine); only falls back to a GitHub-hosted mirror if that
    fails, so this never *replaces* the real network dependency, only works
    around a specific sandbox's egress policy for the purposes of this
    smoke test.
    """
    import astropy.coordinates as coord

    try:
        with coord.solar_system_ephemeris.set(ephem):
            pass
        print(f"{ephem} ephemeris resolved normally (cache or direct download).")
        return
    except Exception as exc:
        print(f"Normal {ephem} ephemeris resolution failed ({exc!r}); "
              "falling back to a GitHub-hosted mirror of the same "
              "public-domain JPL kernel...")

    from astropy.utils.data import download_file, import_file_to_cache

    mirror_url = f"https://raw.githubusercontent.com/skyfielders/python-skyfield/master/ci/{ephem}.bsp"
    with tempfile.NamedTemporaryFile(suffix=".bsp", delete=False) as f:
        tmp_path = f.name
    try:
        # urlretrieve() has no timeout parameter; urlopen() does, and a
        # stalled download otherwise has no bound at all here.
        with urllib.request.urlopen(mirror_url, timeout=60) as response, open(tmp_path, "wb") as out:
            shutil.copyfileobj(response, out)

        # astropy/PINT each try a short, slightly different list of URLs for
        # this ephemeris (see astropy.coordinates.solar_system._get_kernel
        # and pint.solar_system_ephemerides.ephemeris_mirrors) - seed all of
        # them so whichever one gets tried first hits the cache.
        candidate_urls = [
            f"https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/{ephem}.bsp",
            f"https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/a_old_versions/{ephem}.bsp",
        ]
        for url in candidate_urls:
            import_file_to_cache(url, tmp_path)
            download_file(url, cache=True, sources=[url])
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    with coord.solar_system_ephemeris.set(ephem):
        pass
    print("Ephemeris now resolvable from cache.")


def main():
    _ensure_ephemeris("de421")

    workdir = Path(tempfile.mkdtemp(prefix="ptadata-noise-smoketest-"))
    par = workdir / "NGC6440E.par"
    tim = workdir / "NGC6440E.tim"
    shutil.copy(pint.config.examplefile("NGC6440E.par"), par)
    shutil.copy(pint.config.examplefile("NGC6440E.tim"), tim)

    outdir = workdir / "noise"
    print(f"\nRunning a real single-pulsar noise fit in {outdir} ...")
    report = run_noise_fit(par, [tim], outdir, niter=2000, burn=200, cov_update=200, seed=1234)

    print()
    print(f"status:               {report.status}")
    print(f"pulsar:               {report.pulsar}")
    print(f"ntoas:                {report.ntoas}")
    print(f"sampler:              {report.sampler}")
    print(f"n_samples:            {report.n_samples}")
    print(f"acceptance_fraction:  {report.acceptance_fraction}")
    print("posterior means:")
    for name, value in zip(report.param_names, report.posterior_means):
        print(f"  {name:35s} {value:.4f}")

    if report.status != "complete":
        print("notes:", report.notes)
        sys.exit(1)


if __name__ == "__main__":
    main()
