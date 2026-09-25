"""Single-pulsar Bayesian noise fitting: enterprise + PINT + PTMCMCSampler.

Phase 1 scope: a marginalised timing model, per-backend white noise (EFAC +
t2equad, plus optional ECORR), achromatic power-law red noise, and optional
chromatic (DM) power-law noise - all *per pulsar*. There is deliberately no
cross-pulsar common process / gravitational-wave-background signal here -
that's future-phase scope (see the ``asimov_ptadata.noise`` module
docstring). This noise model matters beyond just Phase 1 itself: the
downstream common-process search this feeds (a shared cubic-in-time signal
across the array, i.e. the jerk of the Solar-System barycentre) can only be
told apart from a pulsar's own timing-model/red/DM noise if those are
modelled - and marginalised - properly first.

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

Timing-model marginalisation
------------------------------
``gp_signals.TimingModel(use_svd=True)`` projects the pulsar's own
timing-model design matrix (phase/F0/F1/... - a constant, a linear, and a
quadratic term in time, plus whatever else the ``.par`` file fits for) out
of the likelihood analytically, as an infinite-variance Gaussian process
rather than a set of free parameters to sample. ``use_svd=True`` asks
enterprise to orthogonalise that design matrix via its SVD before use - the
``enterprise`` docstring (``gp_signals.TimingModel``) recommends this "for
better conditioning", and it costs nothing extra here (the design matrix is
tiny compared to the number of TOAs). This is what makes the array-wide
cubic-in-time (jerk) search this pipeline feeds meaningful: without it, a
shared cubic signal would be partially degenerate with, and could leak into,
each pulsar's own unmodelled quadratic (F1) spin-down term.

Per-backend white noise, and how enterprise derives "backend" from an IPTA
DR2-style ``.tim`` file
---------------------------------------------------------------------------
White noise (EFAC, t2equad, and ECORR) is fit **per backend/receiver
combination**, not once per pulsar, using
``selections.Selection(selections.by_backend)``. This is the standard IPTA
practice (different backends have different systematics), and it matters
here specifically because the downstream GWB/jerk search holds these values
fixed: a single averaged-over-backends EFAC would bias the *fixed* per-pulsar
noise level the common-process fit is built on top of, in a way that could
itself imitate or mask a shared low-frequency signal.

What "backend" means here is entirely enterprise's own call, not something
this module decides - read directly from
``enterprise.pulsar.BasePulsar.backend_flags`` (``enterprise/pulsar.py``,
confirmed against the installed ``enterprise-pulsar`` package rather than
assumed): it builds an array of per-TOA backend labels by trying tim-file
flags in *ascending* priority ``fe``+``be`` (combined as ``"fe_be"``), then
``f``, then ``i``, then ``sys``, then ``g``, then ``group`` - each later flag
in that list, if present on a TOA, *overwrites* whatever an earlier one set,
so ``-group`` (if present) wins over everything else, then ``-g``, then
``-sys``, then ``-i``, then ``-f``, and only TOAs with none of those fall
back to a plain ``"<fe>_<be>"`` combination. IPTA DR2-style ``.tim`` files
typically carry both a ``-group`` flag (e.g. ``PuppiL-wide``) and an
``-f``/``-fe``/``-be`` triple; per the priority above, ``-group`` is what
``selections.by_backend`` actually splits on for that data - the finer
``-f`` receiver/backend combination is *not* what gets used for those TOAs,
which is worth being explicit about since it's not obvious from the flag
names alone. ``selections.by_backend`` itself (``enterprise/signals/
selections.py``) is a thin wrapper: it just groups TOA indices by the
unique values of whatever ``backend_flags`` returns.

ECORR (optional, default on)
------------------------------
``white_signals.EcorrKernelNoise`` (also per backend, via the same
selection) models the correlated jitter/scintillation noise between TOAs
from the same observing epoch - real for high-cadence, multi-frequency
backends, and, like EFAC/t2equad, another per-backend noise term that would
bias the fixed noise level handed to the GWB/jerk search if omitted where
it's actually present in the data. Made optional (``use_ecorr``, default
``True``) since not every backend/receiver combination in every data set
actually needs it (e.g. single-TOA-per-epoch backends), and forcing it on
for those can make the sampler work to constrain an unconstrained parameter
for no benefit.

DM (chromatic) red noise (optional, default on)
---------------------------------------------------
A second power-law Fourier-basis GP, built directly from
``gp_signals.BasisGP`` with ``utils.createfourierdesignmatrix_dm`` as its
basis function (rather than the achromatic
``utils.createfourierdesignmatrix_red`` basis ``gp_signals.FourierBasisGP``
hard-codes) - confirmed directly against the installed ``enterprise-pulsar``
source (``enterprise/signals/gp_bases.py``): ``createfourierdesignmatrix_dm``
is already an ``@parameter.function``-wrapped callable exactly like the
achromatic basis function ``FourierBasisGP`` uses internally, it just scales
each Fourier column by ``(fref / freqs) ** 2`` to make the basis
frequency-dependent (chromatic, as DM delays are) rather than
frequency-independent. ``gp_signals.BasisGP`` (the generic building block
``FourierBasisGP`` itself is implemented on top of) takes that basis function
directly, so no other enterprise machinery is needed. This matters for the
same reason DM noise always matters in a real PTA analysis: uncorrected
interstellar-medium DM variations are strongly time-correlated and
frequency-dependent, and can otherwise leak into (or be confused with) both
the achromatic red-noise model and, more importantly for this pipeline, a
shared low-frequency common process. Made optional (``use_dm_noise``,
default ``True``) for narrowband/DM-insensitive data sets where it isn't
identifiable and would just waste sampler time.
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


def _build_noise_model(
    red_noise_components=10,
    dm_noise_components=10,
    use_ecorr=True,
    use_dm_noise=True,
    fixed=False,
    selection_fn=None,
):
    """
    Build the per-pulsar signal model shared by every pulsar in both the
    single-pulsar noise fit (``_build_pta``, ``fixed=False``, every
    parameter a real prior) and the fixed-noise GWB/jerk search
    (``asimov_ptadata.gwb_fit._build_joint_pta``, ``fixed=True``, every
    parameter a non-sampled ``parameter.Constant``): the marginalised
    timing model, per-backend white noise (+ optional ECORR), achromatic
    red noise, and optional DM noise - see this module's docstring for why
    each piece is there. Returns a *signal model*, not yet applied to a
    pulsar - call the result with a pulsar (``model(psr)``) to get that
    pulsar's ``SignalCollection``, exactly like any other ``enterprise``
    signal sum.

    When ``fixed=True``, every noise parameter here is built as
    ``parameter.Constant()`` with no value - the caller is expected to fill
    those in afterwards via ``signal_base.PTA.set_default_params()`` (keyed
    by the model's own full parameter names, e.g.
    ``"<psr>_<backend>_efac"``), rather than baking per-backend fixed values
    into this function's signature. This is deliberate, not incidental:
    building one ``parameter.Constant(value)`` *per backend* and combining
    those into one ``white_signals.MeasurementNoise``/``EcorrKernelNoise``
    signal per backend (rather than the single selection-covering-every-key
    signal built here) was tried first and produces a ``TypeError`` deep in
    ``enterprise.signals.signal_base.ConstantParameter._solve_D1`` (a
    ``float / ShermanMorrison`` operand-type error) as soon as more than one
    backend's ECORR is present - confirmed directly while building this.
    ``set_default_params`` avoids that entirely by keeping exactly the same
    single-signal-per-selection structure the free/sampled case uses,
    differing only in which parameter *class* (``Uniform`` vs ``Constant``)
    it's built from.

    Parameters
    ----------
    red_noise_components, dm_noise_components : int
        Number of Fourier components for the achromatic red-noise and
        (if enabled) DM-noise GPs, respectively.
    use_ecorr : bool
        Whether to include per-backend ECORR (``white_signals.
        EcorrKernelNoise``).
    use_dm_noise : bool
        Whether to include the chromatic DM-noise GP.
    fixed : bool
        ``False`` (default): every noise parameter is a real ``Uniform``
        prior, to be sampled. ``True``: every noise parameter is an
        unset ``parameter.Constant()``, to be filled in by the caller via
        ``PTA.set_default_params()``.
    selection_fn : callable, optional
        The backend-selection function passed to
        ``selections.Selection(...)`` for the white-noise/ECORR signals.
        Defaults to ``selections.by_backend`` (see this module's docstring
        for exactly how that derives "backend" from a pulsar's tim-file
        flags). Overridable so ``gwb_fit._build_joint_pta`` can fall back to
        ``selections.no_selection`` for a *legacy* fixed-noise dict that
        predates per-backend noise support (see that module's docstring).

    Returns
    -------
    An enterprise signal model - the sum of ``gp_signals.TimingModel`` and
    the white/red/DM noise signals described above.
    """
    from enterprise.signals import gp_priors, gp_signals, parameter, selections, white_signals

    if selection_fn is None:
        selection_fn = selections.by_backend
    selection = selections.Selection(selection_fn)

    def _param(lo, hi):
        return parameter.Constant() if fixed else parameter.Uniform(lo, hi)

    model = gp_signals.TimingModel(use_svd=True)

    efac = _param(0.1, 5.0)
    log10_t2equad = _param(-10, -5)
    model += white_signals.MeasurementNoise(efac=efac, log10_t2equad=log10_t2equad, selection=selection)

    if use_ecorr:
        log10_ecorr = _param(-10, -5)
        model += white_signals.EcorrKernelNoise(log10_ecorr=log10_ecorr, selection=selection)

    log10_A = _param(-20, -11)
    gamma = _param(0, 7)
    powerlaw = gp_priors.powerlaw(log10_A=log10_A, gamma=gamma)
    model += gp_signals.FourierBasisGP(powerlaw, components=red_noise_components)

    if use_dm_noise:
        from enterprise.signals import utils

        log10_A_dm = _param(-20, -11)
        gamma_dm = _param(0, 7)
        dm_powerlaw = gp_priors.powerlaw(log10_A=log10_A_dm, gamma=gamma_dm)
        dm_basis = utils.createfourierdesignmatrix_dm(nmodes=dm_noise_components)
        model += gp_signals.BasisGP(dm_powerlaw, dm_basis, name="dm_gp")

    return model


def _build_pta(
    par_file,
    tim_files,
    red_noise_components=10,
    dm_noise_components=10,
    use_ecorr=True,
    use_dm_noise=True,
):
    """
    Build a real, single-pulsar ``enterprise`` PTA likelihood: a
    marginalised timing model, per-backend EFAC + t2equad white noise
    (+ optional ECORR), a power-law Fourier-basis GP for achromatic red
    noise, and (optionally) a chromatic power-law GP for DM noise. No
    common/GWB process - single-pulsar scope only. See this module's
    docstring for the reasoning behind each piece.

    Parameters
    ----------
    par_file : str or Path
        Path to the (already QC'd) timing model.
    tim_files : str, Path, or list of these
        Path(s) to the (already QC'd) TOA file(s).
    red_noise_components, dm_noise_components : int
        Number of Fourier components for the red-noise and (if enabled)
        DM-noise GPs.
    use_ecorr : bool
        Whether to include per-backend ECORR.
    use_dm_noise : bool
        Whether to include the chromatic DM-noise GP.

    Returns
    -------
    (enterprise.pulsar.Pulsar, enterprise.signals.signal_base.PTA)
    """
    from enterprise.pulsar import Pulsar
    from enterprise.signals import signal_base

    tim_files = [tim_files] if isinstance(tim_files, (str, Path)) else list(tim_files)
    tim_arg = [str(t) for t in tim_files] if len(tim_files) > 1 else str(tim_files[0])

    psr = Pulsar(str(par_file), tim_arg, timing_package="pint")

    model = _build_noise_model(
        red_noise_components=red_noise_components,
        dm_noise_components=dm_noise_components,
        use_ecorr=use_ecorr,
        use_dm_noise=use_dm_noise,
        fixed=False,
    )
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
    dm_noise_components=10,
    use_ecorr=True,
    use_dm_noise=True,
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
    red_noise_components, dm_noise_components : int
        Number of red-noise / (if enabled) DM-noise Fourier components.
    use_ecorr : bool
        Whether to include per-backend ECORR (default ``True``).
    use_dm_noise : bool
        Whether to include the chromatic DM-noise GP (default ``True``).
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
        psr, pta = _build_pta(
            par_file,
            tim_files,
            red_noise_components=red_noise_components,
            dm_noise_components=dm_noise_components,
            use_ecorr=use_ecorr,
            use_dm_noise=use_dm_noise,
        )
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
