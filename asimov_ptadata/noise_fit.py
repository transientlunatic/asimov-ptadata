"""Single-pulsar Bayesian noise fitting: enterprise + PINT + PTMCMCSampler.

Phase 1 walking-skeleton scope: an independent white-noise (EFAC + t2equad)
plus power-law red-noise model *per pulsar*. There is deliberately no
cross-pulsar common process / gravitational-wave-background signal here -
that's future-phase scope (see the ``asimov_ptadata.noise`` module
docstring).

This mirrors ``asimov_ptadata.reduce``'s split between "real science logic"
(this module) and "Asimov plumbing" (``asimov_ptadata.noise``): nothing here
imports ``asimov``, and - importantly - nothing at *module import time*
imports ``enterprise`` or ``PTMCMCSampler`` either. Those are both pulled in
lazily inside ``run_noise_fit``/``_build_pta`` so that importing this module
(e.g. transitively, via the CLI) never requires the heavier ``noise`` extra
to be installed just to run ``ptadata reduce``.

Sampler choice
---------------
``PTMCMCSampler`` (the IPTA-standard parallel-tempering MCMC sampler used
throughout the ``enterprise``/``enterprise_extensions`` ecosystem) is used
here rather than a general-purpose alternative like ``emcee`` or ``dynesty``.
It installed cleanly from PyPI (``pip install ptmcmcsampler``) alongside
``enterprise-pulsar`` with no native-build pain, so there was no need to
fall back to anything else.

Timing backend
---------------
``enterprise.pulsar.Pulsar(..., timing_package="pint")`` is used explicitly
so this never depends on ``libstempo`` (not installed, and not part of this
project's dependency stack - ``pint-pulsar`` is already a base dependency of
``asimov-ptadata`` via the reduce pipeline).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import yaml


@dataclasses.dataclass
class NoiseFitReport:
    pulsar: str
    ntoas: int
    param_names: list
    posterior_means: list
    n_samples: int
    acceptance_fraction: float | None
    sampler: str
    status: str
    notes: list

    def save(self, path):
        with open(path, "w") as f:
            yaml.safe_dump(dataclasses.asdict(self), f, sort_keys=False)


def _build_pta(par_file, tim_files, red_noise_components=10):
    """
    Build a real, minimal single-pulsar ``enterprise`` PTA likelihood:
    EFAC + t2equad white noise, plus a power-law Fourier-basis GP for red
    noise. No common/GWB process - single-pulsar scope only.

    Parameters
    ----------
    par_file : str or Path
        Path to the (already QC'd) timing model.
    tim_files : str, Path, or list of these
        Path(s) to the (already QC'd) TOA file(s).
    red_noise_components : int
        Number of Fourier components for the red-noise GP.

    Returns
    -------
    (enterprise.pulsar.Pulsar, enterprise.signals.signal_base.PTA)
    """
    from enterprise.pulsar import Pulsar
    from enterprise.signals import gp_priors, gp_signals, parameter, selections, signal_base, white_signals

    tim_files = [tim_files] if isinstance(tim_files, (str, Path)) else list(tim_files)
    tim_arg = [str(t) for t in tim_files] if len(tim_files) > 1 else str(tim_files[0])

    psr = Pulsar(str(par_file), tim_arg, timing_package="pint")

    selection = selections.Selection(selections.no_selection)

    efac = parameter.Uniform(0.1, 5.0)
    log10_t2equad = parameter.Uniform(-10, -5)
    white = white_signals.MeasurementNoise(efac=efac, log10_t2equad=log10_t2equad, selection=selection)

    log10_A = parameter.Uniform(-20, -11)
    gamma = parameter.Uniform(0, 7)
    powerlaw = gp_priors.powerlaw(log10_A=log10_A, gamma=gamma)
    red_noise = gp_signals.FourierBasisGP(powerlaw, components=red_noise_components)

    model = white + red_noise
    pta = signal_base.PTA([model(psr)])
    return psr, pta


def run_noise_fit(
    par_file,
    tim_files,
    outdir,
    niter=6000,
    burn=1000,
    cov_update=None,
    red_noise_components=10,
    seed=None,
):
    """
    Run a real single-pulsar noise fit and write the chain plus a summary
    report to ``outdir``.

    Parameters
    ----------
    par_file, tim_files : see :func:`_build_pta`
    outdir : str or Path
        Directory to write ``noise_report.yml`` and the ``chain/`` directory
        (PTMCMCSampler's own chain files) into.
    niter : int
        Total number of sampler iterations.
    burn : int
        Number of burn-in iterations, discarded from the posterior mean.
    cov_update : int, optional
        PTMCMCSampler's ``covUpdate`` (how often the adaptive-metropolis
        proposal covariance is recomputed). Defaults to ``burn``. This
        *must* match ``burn`` or PTMCMCSampler's own DE-buffer bookkeeping
        raises a ``ValueError`` partway through the run (its internal
        AM-proposal buffer is sized from ``covUpdate``, its DE-jump buffer
        from ``burn``, and ``_updateDEbuffer`` assumes the two match) -
        confirmed the hard way while building this pipeline.
    red_noise_components : int
        Number of red-noise Fourier components.
    seed : int, optional
        Seed for the initial-sample RNG, for reproducibility.

    Returns
    -------
    NoiseFitReport
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if cov_update is None:
        cov_update = burn
    elif cov_update != burn:
        report = NoiseFitReport(
            pulsar=Path(par_file).stem,
            ntoas=0,
            param_names=[],
            posterior_means=[],
            n_samples=0,
            acceptance_fraction=None,
            sampler="PTMCMCSampler",
            status="failed",
            notes=[
                f"cov_update ({cov_update}) must match burn ({burn}) - "
                "PTMCMCSampler's DE-jump buffer (sized from burn) and its "
                "AM-proposal buffer (sized from covUpdate) must agree, or "
                "_updateDEbuffer raises a ValueError partway through "
                "sampling. Failing fast here instead of letting that happen."
            ],
        )
        report.save(outdir / "noise_report.yml")
        return report

    pulsar_name = Path(par_file).stem
    notes = []

    try:
        psr, pta = _build_pta(par_file, tim_files, red_noise_components=red_noise_components)
    except Exception as exc:
        report = NoiseFitReport(
            pulsar=pulsar_name,
            ntoas=0,
            param_names=[],
            posterior_means=[],
            n_samples=0,
            acceptance_fraction=None,
            sampler="PTMCMCSampler",
            status="failed",
            notes=[f"failed to build PTA: {exc}"],
        )
        report.save(outdir / "noise_report.yml")
        return report

    from PTMCMCSampler.PTMCMCSampler import PTSampler

    rng = np.random.default_rng(seed)
    x0 = np.hstack([p.sample() for p in pta.params]) if seed is None else np.hstack(
        [_sample_with_rng(p, rng) for p in pta.params]
    )
    ndim = len(x0)
    cov = np.diag(np.ones(ndim) * 0.1**2)

    chain_dir = outdir / "chain"
    try:
        sampler = PTSampler(ndim, pta.get_lnlikelihood, pta.get_lnprior, cov, outDir=str(chain_dir))
        sampler.sample(
            x0,
            niter,
            burn=burn,
            thin=1,
            isave=max(burn, 1),
            covUpdate=cov_update,
            SCAMweight=30,
            AMweight=15,
            DEweight=50,
        )
        chain = np.loadtxt(chain_dir / "chain_1.txt")
    except Exception as exc:
        # A report has to land here regardless of outcome: the Asimov
        # pipeline's detect_completion() just checks for this file's
        # existence, so a bare exception here (an unstable proposal, a
        # malformed/truncated chain file, ...) would otherwise leave the
        # job polling forever instead of surfacing a visible failure -
        # same reasoning as the _build_pta failure path above.
        report = NoiseFitReport(
            pulsar=pulsar_name,
            ntoas=0,
            param_names=[],
            posterior_means=[],
            n_samples=0,
            acceptance_fraction=None,
            sampler="PTMCMCSampler",
            status="failed",
            notes=[f"sampling failed: {exc}"],
        )
        report.save(outdir / "noise_report.yml")
        return report

    if chain.ndim == 1:
        chain = chain.reshape(1, -1)

    post_burn = chain[burn:, :ndim] if chain.shape[0] > burn else chain[:, :ndim]
    posterior_means = np.mean(post_burn, axis=0).tolist() if len(post_burn) else []

    if chain.shape[0] > 1:
        moved = np.any(np.diff(chain[:, :ndim], axis=0) != 0, axis=1)
        acceptance_fraction = float(np.mean(moved))
    else:
        acceptance_fraction = None

    report = NoiseFitReport(
        pulsar=pulsar_name,
        ntoas=int(len(psr.toas)),
        param_names=list(pta.param_names),
        posterior_means=posterior_means,
        n_samples=int(chain.shape[0]),
        acceptance_fraction=acceptance_fraction,
        sampler="PTMCMCSampler",
        status="complete",
        notes=notes,
    )
    report.save(outdir / "noise_report.yml")
    return report


def _sample_with_rng(param, rng):
    """Sample an enterprise Parameter using a specific numpy Generator, for reproducible seeding."""
    # enterprise Parameter.sample() ultimately calls np.random under the
    # hood without accepting a generator, so seed the legacy global state
    # for this one draw rather than depending on private enterprise internals.
    state = np.random.get_state()
    try:
        np.random.seed(int(rng.integers(0, 2**31 - 1)))
        return param.sample()
    finally:
        np.random.set_state(state)
