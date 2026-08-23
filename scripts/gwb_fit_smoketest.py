#!/usr/bin/env python3
"""Standalone smoke test: a real, joint two-pulsar fixed-noise GWB
common-process fit (enterprise + PINT + PTMCMCSampler), run through
``asimov_ptadata.gwb_fit.run_gwb_fit`` - the same function the
``ptadata-gwb`` Asimov pipeline's HTCondor job runs via ``ptadata gwb-run``.

Mirrors ``scripts/noise_fit_smoketest.py``'s ephemeris-fallback handling (see
that script's docstring for the full explanation) and, deliberately, is kept
out of the hermetic pytest suite for the same reason: a real fit needs a
solar-system ephemeris that PINT resolves over the network the first time.

Fixture pulsars
----------------
Two genuinely distinct real pulsars, both bundled with PINT's own example
data (not a single pulsar's TOAs re-used under two aliases): ``NGC6440E``
(renamed ``1748-2021E``, the same fixture Phase 1 uses) and
``J1028-5819-example`` (kept under its own name, ``J1028-5819`` - a clean,
similarly small, binary-free TEMPO2-format example PINT ships specifically
for tests, converted from ``tempo2``'s own ``example1.par``/``.tim``). Using
two real, independent pulsars (rather than a duplicated-TOA stand-in) makes
the Hellings-Downs common-process construction in
``asimov_ptadata.gwb_fit._build_joint_pta`` structurally real: two distinct
sky positions with two distinct TOA sets, jointly constraining one shared
GWB parameter pair.

Usage
-----
    python scripts/gwb_fit_smoketest.py
"""

import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pint.config  # noqa: E402

from asimov_ptadata.gwb_fit import run_gwb_fit  # noqa: E402

# name -> the PINT-bundled example file stem to copy/rename it from.
FIXTURE_PULSARS = {
    "1748-2021E": "NGC6440E",
    "J1028-5819": "J1028-5819-example",
}

# Fixed per-pulsar noise values this smoke test holds constant, standing in
# for Phase 1 (`ptadata-noise`) posterior means - plausible values in the
# same ranges noise_fit.py samples over, not fit results from a real
# noise-fit run (this script only exercises the GWB stage in isolation).
FIXED_NOISE = {
    "1748-2021E": {"efac": 1.1, "log10_t2equad": -7.0, "red_noise_gamma": 3.5, "red_noise_log10_A": -14.0},
    "J1028-5819": {"efac": 0.9, "log10_t2equad": -7.5, "red_noise_gamma": 2.8, "red_noise_log10_A": -14.5},
}


def _ensure_ephemeris(ephem="de421"):
    """See scripts/noise_fit_smoketest.py's docstring for the full story;
    identical fallback, duplicated here rather than imported since these are
    both standalone, no-package-dependency scripts."""
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
        with urllib.request.urlopen(mirror_url, timeout=60) as response, open(tmp_path, "wb") as out:
            shutil.copyfileobj(response, out)

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

    workdir = Path(tempfile.mkdtemp(prefix="ptadata-gwb-smoketest-"))
    pulsars = []
    for name, stem in FIXTURE_PULSARS.items():
        par = workdir / f"{name}.par"
        tim = workdir / f"{name}.tim"
        shutil.copy(pint.config.examplefile(f"{stem}.par"), par)
        shutil.copy(pint.config.examplefile(f"{stem}.tim"), tim)
        pulsars.append({"name": name, "par": str(par), "tim": [str(tim)], **FIXED_NOISE[name]})

    outdir = workdir / "gwb"
    print(f"\nRunning a real {len(pulsars)}-pulsar joint fixed-noise GWB fit in {outdir} ...")
    report = run_gwb_fit(pulsars, outdir, niter=2000, burn=200, cov_update=200, seed=1234)

    print()
    print(f"status:               {report.status}")
    print(f"pulsars:              {report.pulsars}")
    print(f"ntoas:                {report.ntoas}")
    print(f"sampler:              {report.sampler}")
    print(f"n_samples:            {report.n_samples}")
    print(f"acceptance_fraction:  {report.acceptance_fraction}")
    print("posterior means (shared GWB parameters):")
    for name, value in zip(report.param_names, report.posterior_means):
        print(f"  {name:35s} {value:.4f}")

    if report.status != "complete":
        print("notes:", report.notes)
        sys.exit(1)


if __name__ == "__main__":
    main()
