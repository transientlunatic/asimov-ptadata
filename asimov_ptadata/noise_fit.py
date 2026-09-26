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
    # Filled by the two-stage fit (see run_noise_fit): per-parameter
    # convergence diagnostics for the red/DM noise parameters the
    # downstream fixed-noise searches depend on, and the overall verdict.
    convergence: dict | None = None
    converged: bool | None = None

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


# ---------------------------------------------------------------------------
# Sampling helpers: parameter groups, a data-driven starting point, and
# convergence diagnostics.
# ---------------------------------------------------------------------------

_WHITE_SUFFIXES = ("_efac", "_log10_t2equad", "_log10_ecorr")


def _is_white(name):
    return name.endswith(_WHITE_SUFFIXES)


def _backend_groups(names):
    """Indices of each backend's white-noise parameters, one list per backend."""
    backends = {}
    for i, name in enumerate(names):
        for suffix in _WHITE_SUFFIXES:
            if name.endswith(suffix):
                backends.setdefault(name[: -len(suffix)], []).append(i)
    return list(backends.values())


def _param_groups(names):
    """PTMCMC jump groups: every parameter; each backend's white-noise
    parameters; the red-noise pair; the DM-noise pair; red+DM together; and
    all white-noise parameters.

    Without groups, every proposal moves all parameters at once - 121 of them
    for J1713+0747 in IPTA DR2 - and the chain can sit for 10^5 iterations
    far from the likelihood peak (it did: its red/DM noise stayed ~4 dex too
    low, dlnL ~ -400). Per-group jumps let the few red/DM hyperparameters
    move on their own.
    """
    groups = [list(range(len(names)))]
    backends = _backend_groups(names)
    groups += backends
    red = [i for i, n in enumerate(names) if "red_noise" in n]
    dm = [i for i, n in enumerate(names) if "dm_gp" in n]
    groups += [g for g in (red, dm, red + dm) if g]
    white = sorted(i for idx in backends for i in idx)
    if white and len(white) < len(names):
        groups.append(white)
    return groups


def _bounds(param):
    defaults = getattr(getattr(param, "prior", None), "_defaults", {}) or {}
    if "pmin" in defaults and "pmax" in defaults:
        return float(defaults["pmin"]), float(defaults["pmax"])
    draws = np.array([param.sample() for _ in range(200)], dtype=float)
    return float(draws.min()), float(draws.max())


def _initial_point(pta, x_prior, rounds=2, maxfev=400, maxfev_backend=60):
    """A starting point near the likelihood peak.

    White noise starts at nominal values (EFAC 1, EQUAD/ECORR near the
    bottom of their priors), then ``rounds`` of coordinate ascent alternate
    between the red/DM hyperparameters (one bounded Powell maximisation) and
    each backend's white-noise parameters in turn, followed by a final
    red/DM pass. Both halves matter: starting the chain from a prior draw
    left J1713+0747's red/DM noise ~4 dex too low (dlnL ~ -400), while
    optimising only red/DM with white noise at its floor let a flat
    (gamma ~ 0.3) "red" process stand in for the missing white noise.
    """
    from scipy.optimize import minimize

    names = pta.param_names
    x = np.array(x_prior, dtype=float)
    for i, (name, param) in enumerate(zip(names, pta.params)):
        lo, hi = _bounds(param)
        if name.endswith("_efac"):
            x[i] = min(max(1.0, lo), hi)
        elif _is_white(name):
            x[i] = lo + 0.05 * (hi - lo)
        elif name.endswith("log10_A"):
            x[i] = lo + 0.6 * (hi - lo)
        else:
            x[i] = 0.5 * (lo + hi)

    def maximise(idx, fev):
        bounds = [_bounds(pta.params[i]) for i in idx]

        def neg_lnpost(sub):
            y = x.copy()
            y[idx] = sub
            lp = pta.get_lnprior(y)
            return np.inf if not np.isfinite(lp) else -(pta.get_lnlikelihood(y) + lp)

        result = minimize(neg_lnpost, x[idx], method="Powell", bounds=bounds, options={"maxfev": fev, "xtol": 1e-2})
        if np.isfinite(result.fun):
            x[idx] = result.x

    red_dm = [i for i, n in enumerate(names) if not _is_white(n)]
    backends = _backend_groups(names)
    for _ in range(rounds):
        if red_dm:
            maximise(red_dm, maxfev)
        for group in backends:
            maximise(group, maxfev_backend)
    if red_dm:
        maximise(red_dm, maxfev)
    return x


def effective_sample_size(x):
    """Effective sample size of a 1-D chain from its autocorrelation,
    summed until it first drops below 0.05."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    # ptp, not var: a flat chain of e.g. 3.14 has var ~1e-30, not 0.
    if n < 4 or np.ptp(x) == 0:
        return float(n)
    y = x - x.mean()
    f = np.fft.rfft(y, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n] / (np.var(x) * n)
    tau = 1.0
    for k in range(1, n):
        if acf[k] < 0.05:
            break
        tau += 2.0 * acf[k]
    return float(n / tau)


def split_shift(x):
    """|mean(first half) - mean(second half)| in units of the chain's SD."""
    x = np.asarray(x, dtype=float)
    sd = np.std(x)
    if len(x) < 4 or np.ptp(x) == 0:
        return 0.0
    h = len(x) // 2
    return float(abs(x[:h].mean() - x[h:].mean()) / sd)


def convergence_summary(chain, names, min_ess=200.0, max_split_shift=0.3):
    """Per-parameter ESS and split-half shift, and whether every parameter
    passes both thresholds."""
    per_param = {}
    ok = True
    for i, name in enumerate(names):
        ess = effective_sample_size(chain[:, i])
        shift = split_shift(chain[:, i])
        per_param[name] = {"ess": round(ess, 1), "split_shift": round(shift, 3)}
        ok = ok and ess >= min_ess and shift <= max_split_shift
    return {"min_ess": min_ess, "max_split_shift": max_split_shift, "parameters": per_param}, bool(ok)


# Initial proposal widths by parameter type (the adaptive proposals take
# over from these); one 0.1 for everything was far too wide for EFACs and
# too narrow for spectral indices.
_PROPOSAL_SCALES = (
    ("_efac", 0.05),
    ("_log10_t2equad", 0.2),
    ("_log10_ecorr", 0.2),
    ("_gamma", 0.3),
    ("_log10_A", 0.2),
)


def _proposal_scales(names):
    return np.array([next((w for suffix, w in _PROPOSAL_SCALES if n.endswith(suffix)), 0.1) for n in names])


def _run_ptmcmc(logl, logp, x0, groups, outdir, niter, adapt, names=None):
    """Run PTMCMCSampler, re-estimating its proposal covariance every
    ``adapt`` iterations.

    PTMCMCSampler's own ``burn`` sizes its DE-jump buffer and must equal
    ``covUpdate`` (otherwise ``_updateDEbuffer`` raises partway through), so
    both are set to ``adapt`` here. That is unrelated to how many samples the
    caller discards: tying adaptation to the discarded burn-in (5000 of
    20000) meant the covariance was re-estimated only four times per run,
    and stage 1's white noise mixed badly (ESS ~10-30).
    """
    from PTMCMCSampler.PTMCMCSampler import PTSampler

    ndim = len(x0)
    scales = _proposal_scales(names) if names is not None else np.full(ndim, 0.1)
    cov = np.diag(scales**2)
    sampler = PTSampler(ndim, logl, logp, cov, groups=groups, outDir=str(outdir), verbose=False)
    sampler.sample(
        x0, niter, burn=adapt, thin=1, isave=max(adapt, 1), covUpdate=adapt,
        SCAMweight=30, AMweight=15, DEweight=50,
    )
    chain = np.loadtxt(Path(outdir) / "chain_1.txt")
    return chain.reshape(1, -1) if chain.ndim == 1 else chain


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
    two_stage=True,
    stage2_niter=None,
    optimise_start=True,
    min_ess=200.0,
    max_split_shift=0.3,
):
    """
    Run a real single-pulsar noise fit and write the chain plus a summary
    report to ``outdir``.

    With ``two_stage`` (the default) this is:

    1. **All parameters**, from a starting point near the likelihood peak for
       the red/DM noise (``optimise_start``, see :func:`_initial_point`),
       with per-group jumps (:func:`_param_groups`). White-noise posterior
       means are taken from here.
    2. **Red- and DM-noise hyperparameters only**, with the white noise
       fixed at those means, for ``stage2_niter`` iterations (default
       ``niter``). Four parameters mix far better than the ~40-120 of the
       joint model, and these are the values the fixed-noise GWB and jerk
       searches depend on most.

    ``convergence``/``converged`` in the report come from the chain that
    determined the red/DM values (stage 2, or stage 1 if ``two_stage`` is
    off): every one of those parameters must reach an effective sample size
    of at least ``min_ess`` and a split-half mean shift of at most
    ``max_split_shift`` posterior SDs. Stage 1's log-likelihood must also be
    stationary (split-half shift at most ``max_split_shift``): a stage 1
    still climbing towards the peak leaves white-noise values that stage 2
    would otherwise converge around (J1713+0747 did exactly that). The
    ``ptadata-noise`` Asimov pipeline only auto-approves converged fits.

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
        How often (iterations) the adaptive proposal covariance is
        re-estimated; defaults to ``min(burn, 1000)``. Independent of
        ``burn``, which only sets how many samples are discarded - see
        :func:`_run_ptmcmc` for how PTMCMCSampler's own burn/covUpdate
        constraint is met.
    red_noise_components, dm_noise_components : int
        Number of red-noise / (if enabled) DM-noise Fourier components.
    use_ecorr : bool
        Whether to include per-backend ECORR (default ``True``).
    use_dm_noise : bool
        Whether to include the chromatic DM-noise GP (default ``True``).
    seed : int, optional
        Seed for the initial-sample RNG, for reproducibility.
    two_stage, stage2_niter, optimise_start, min_ess, max_split_shift
        See above.

    Returns
    -------
    NoiseFitReport
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if cov_update is None:
        cov_update = max(1, min(burn, 1000))

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

    rng = np.random.default_rng(seed)
    x0 = np.hstack([p.sample() for p in pta.params]) if seed is None else np.hstack(
        [_sample_with_rng(p, rng) for p in pta.params]
    )
    names = list(pta.param_names)
    ndim = len(x0)
    red_dm = [i for i, n in enumerate(names) if not _is_white(n)]

    try:
        if optimise_start:
            x0 = _initial_point(pta, x0)
            notes.append(
                "stage 1 started from optimised red/DM noise: "
                + ", ".join(f"{names[i].split('_', 1)[1]}={x0[i]:.2f}" for i in red_dm)
            )
        chain1 = _run_ptmcmc(
            pta.get_lnlikelihood, pta.get_lnprior, x0, _param_groups(names), outdir / "chain", niter,
            cov_update, names=names,
        )
        post1 = chain1[burn:, :ndim] if chain1.shape[0] > burn else chain1[:, :ndim]
        means = post1.mean(axis=0)
        # Stage 1 must have stopped climbing: a chain still heading for the
        # peak gives white-noise means stage 2 would then converge around.
        lnl1 = chain1[burn:, ndim + 1] if chain1.shape[0] > burn else chain1[:, ndim + 1]
        stage1_lnl_shift = split_shift(lnl1)
        decisive, decisive_names, decisive_chain = post1[:, red_dm], [names[i] for i in red_dm], chain1

        if two_stage and red_dm and len(red_dm) < ndim:
            n2 = int(stage2_niter or niter)
            burn2 = min(burn, max(n2 // 10, 1))
            fixed = means.copy()

            def full(sub):
                y = fixed.copy()
                y[red_dm] = sub
                return y

            k = len(red_dm)
            groups2 = [list(range(k))]
            groups2 += [g for g in (
                [j for j, i in enumerate(red_dm) if "red_noise" in names[i]],
                [j for j, i in enumerate(red_dm) if "dm_gp" in names[i]],
            ) if g and len(g) < k]
            chain2 = _run_ptmcmc(
                lambda sub: pta.get_lnlikelihood(full(sub)),
                lambda sub: pta.get_lnprior(full(sub)),
                means[red_dm], groups2, outdir / "chain_stage2", n2, min(cov_update, burn2),
                names=[names[i] for i in red_dm],
            )
            post2 = chain2[burn2:, :k] if chain2.shape[0] > burn2 else chain2[:, :k]
            means[red_dm] = post2.mean(axis=0)
            decisive, decisive_chain = post2, chain2[:, :k]
            notes.append(f"stage 2: red/DM noise re-sampled for {n2} iterations with white noise fixed")
        else:
            decisive_chain = chain1[:, :ndim]
    except Exception as exc:
        # A report has to land here regardless of outcome: the Asimov
        # pipeline's detect_completion() just checks for this file's
        # existence, so a bare exception here (an unstable proposal, a
        # malformed/truncated chain file, a failed optimisation, ...) would
        # otherwise leave the job polling forever instead of surfacing a
        # visible failure - same reasoning as the _build_pta failure path.
        report = NoiseFitReport(
            pulsar=pulsar_name,
            ntoas=0,
            param_names=[],
            posterior_means=[],
            n_samples=0,
            acceptance_fraction=None,
            sampler="PTMCMCSampler",
            status="failed",
            notes=notes + [f"sampling failed: {exc}"],
        )
        report.save(outdir / "noise_report.yml")
        return report

    posterior_means = means.tolist()
    if decisive_chain.shape[0] > 1:
        moved = np.any(np.diff(decisive_chain, axis=0) != 0, axis=1)
        acceptance_fraction = float(np.mean(moved))
    else:
        acceptance_fraction = None
    convergence, converged = convergence_summary(decisive, decisive_names, min_ess, max_split_shift)
    if not converged:
        notes.append(
            f"not converged: some red/DM noise parameter has ESS < {min_ess} or a split-half shift "
            f"> {max_split_shift} SD"
        )
    convergence["stage 1 lnlikelihood split_shift"] = round(stage1_lnl_shift, 3)
    if stage1_lnl_shift > max_split_shift:
        converged = False
        notes.append(
            f"stage 1 not stationary: its log-likelihood shifted by {stage1_lnl_shift:.2f} SD between halves "
            "(more burn-in or iterations needed)"
        )

    report = NoiseFitReport(
        pulsar=pulsar_name,
        ntoas=int(len(psr.toas)),
        param_names=list(pta.param_names),
        posterior_means=posterior_means,
        n_samples=int(decisive_chain.shape[0]),
        acceptance_fraction=acceptance_fraction,
        sampler="PTMCMCSampler",
        status="complete",
        notes=notes,
        convergence=convergence,
        converged=converged,
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
