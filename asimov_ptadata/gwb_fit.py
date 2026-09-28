"""Fixed-noise, array-wide gravitational-wave-background (GWB) / common
cubic-in-time (jerk) search: enterprise + PINT + PTMCMCSampler.

Phase 2 scope, building directly on ``asimov_ptadata.noise_fit`` (Phase 1):
every pulsar's *entire* non-timing-model noise model (per-backend EFAC +
t2equad, optional per-backend ECORR, achromatic red noise, and optional DM
noise - whatever ``noise_fit._build_pta`` actually sampled for that pulsar)
is held **fixed** at its Phase-1 single-pulsar posterior means, the timing
model is (as in Phase 1) marginalised rather than fixed or sampled, and only
the array-wide common process shared across the whole array is sampled.
Full hierarchical re-fitting of per-pulsar noise jointly with the common
process is deliberately out of scope for this phase - see the
``asimov_ptadata.gwb`` module docstring for the roadmap.

This mirrors ``asimov_ptadata.noise_fit``'s split between "real science
logic" (this module) and "Asimov plumbing" (``asimov_ptadata.gwb``): nothing
here imports ``asimov``, and nothing at *module import time* imports
``enterprise`` or ``PTMCMCSampler`` either - both are pulled in lazily inside
``run_gwb_fit``/``_build_joint_pta`` for the same reason ``noise_fit.py``
does: so importing this module never requires the heavier ``noise`` extra
just to run ``ptadata reduce``.

Building a shared parameter across pulsars
-------------------------------------------
The one part of this module's ``enterprise`` usage that isn't a direct
copy-paste of ``noise_fit.py``'s per-pulsar pattern: a *shared* GWB
amplitude/spectral-index pair, sampled jointly across every pulsar in the
array rather than once per pulsar.

``enterprise.signals.parameter.Parameter.__call__`` special-cases an
*already-instantiated* ``Parameter`` object by returning itself unchanged
(see ``enterprise/signals/parameter.py``) instead of re-instantiating it
under whatever name the enclosing ``Signal`` construction would otherwise
give it. So instantiating ``log10_A_gw``/``gamma_gw`` exactly *once*, with an
explicit shared name, and reusing that same Python object in the
``gp_priors.powerlaw(...)`` call for every pulsar's
``gp_signals.FourierBasisCommonGP`` is what makes
``enterprise.signals.signal_base.PTA.params`` collapse the per-pulsar copies
down to a single shared parameter pair - confirmed directly against a real
two-pulsar PTA while building this: ``PTA.param_names`` came back as exactly
``['gwb_gamma', 'gwb_log10_A']``, not four per-pulsar-prefixed names.

``enterprise.signals.utils.hd_orf`` is an enterprise ``@function``-wrapped
callable (a class factory), not a plain function - it has to be
*instantiated* (called with no arguments, ``utils.hd_orf()``) before being
handed to ``gp_signals.FourierBasisCommonGP`` as its ``orf=`` argument,
exactly like the ``spectrum`` argument. Passing the bare function raises
``AttributeError: type object 'Function' has no attribute '_params'`` deep
inside ``gp_signals.BasisCommonGP.__init__`` - confirmed the hard way while
building this.

``enterprise_extensions`` was deliberately not used to build this signal
(``enterprise_extensions.model_utils``/``models.py`` has ready-made
``common_red_noise_block`` helpers that do the same thing), matching
``noise_fit.py``'s existing choice to build directly from
``enterprise.signals`` building blocks and avoid that package's
healpy/scikit-learn dependency chain.

Fixed noise parameters, and why they're set via ``PTA.set_default_params``
-----------------------------------------------------------------------------
``enterprise.signals.parameter.Constant()`` (not ``Uniform``) is the correct
``enterprise`` API for a non-sampled fixed parameter - it's a class factory
exactly like ``Uniform``/``Normal``, so it plugs directly into
``white_signals.MeasurementNoise``/``gp_priors.powerlaw`` the same way
``Uniform(...)`` does in ``noise_fit.py``. Confirmed directly (a
``parameter.Constant``-only PTA builds and evaluates its likelihood/prior
with zero free per-pulsar-noise parameters) while building this.

The per-pulsar signal model itself - ``noise_fit._build_noise_model(...,
fixed=True)`` - is built with ``fixed=True`` but is otherwise *structurally
identical* to Phase 1's free/sampled model: the same single
``selections.Selection(selections.by_backend)``-selected
``MeasurementNoise``/``EcorrKernelNoise`` signals covering every backend at
once, just built from unset ``parameter.Constant()`` instead of
``parameter.Uniform(...)``. The actual fixed values are then filled in
*after* the ``PTA`` is built, via ``signal_base.PTA.set_default_params()``
(keyed by each parameter's real full name, e.g.
``"<psr>_<backend>_efac"``) - not by constructing a separate
``parameter.Constant(value)`` per backend and combining several
single-backend-selection signals together, which was tried first: with more
than one backend's ECORR present, that raises ``TypeError: unsupported
operand type(s) for /: 'float' and 'ShermanMorrison'`` deep inside
``enterprise.signals.signal_base.ConstantParameter._solve_D1`` - confirmed
directly while building this (see ``noise_fit._build_noise_model``'s own
docstring for the same note, since that's where the shared model-building
code actually lives). ``set_default_params`` sidesteps the problem
entirely, and is in any case the API ``enterprise`` itself documents for
this (its own ``parameter.Constant`` docstring: "Leave ``val=None`` to set
value later, ... with ``signal_base.PTA.set_default_params()``").

Carrying *all* of a pulsar's fixed noise values, not a hard-coded four
------------------------------------------------------------------------
Earlier versions of this module accepted exactly four fixed values per
pulsar (``FIXED_NOISE_PARAMS`` below, kept only for backwards
compatibility - see its own docstring). Now that Phase 1 fits a
pulsar-dependent *set* of noise parameters (a different number of backends
per pulsar, ECORR/DM noise each independently optional), that fixed shape
no longer fits: each ``pulsars`` entry instead carries a
``"noise_params"`` dict of *every* non-timing-model parameter name (with
the ``"<psr>_"`` prefix already stripped, exactly as ``asimov_ptadata.gwb.
_resolve_subject_data`` produces it from a Phase-1 ``noise_report.yml``'s
``param_names``/``posterior_means``) mapped to its Phase-1 posterior mean.
This module doesn't need to know in advance which backends or optional
signals a given pulsar has - it builds the same model structure Phase 1
would have (deriving backend keys from the pulsar's own ``.tim`` data, via
the same ``selections.by_backend`` used in Phase 1) and lets
``set_default_params`` fail loudly (an ``enterprise``-logged "not set!"
warning, then a ``TypeError``/``AttributeError`` from the likelihood trying
to use a ``None`` value) if ``noise_params`` doesn't actually cover
everything that model needs - rather than silently accepting a
partially-specified fixed-noise dict.

Sampler choice and the ``covUpdate``/``burn`` requirement
-----------------------------------------------------------
Same ``PTMCMCSampler`` choice as ``noise_fit.py``, for the same reasons (see
that module's docstring), and the same real bug applies here too:
``covUpdate`` *must* match ``burn`` or PTMCMCSampler's own DE-buffer
bookkeeping raises a ``ValueError`` partway through the run (its internal
AM-proposal buffer is sized from ``covUpdate``, its DE-jump buffer from
``burn``, and ``_updateDEbuffer`` assumes the two match) - this isn't
rediscovered here, just applied again.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import yaml

from .noise_fit import _sample_with_rng

# The exact four fixed-noise parameter names this module used to require -
# from before per-backend white noise, ECORR and DM noise existed, when
# every pulsar's noise model was a single global EFAC + t2equad (no backend
# selection) plus achromatic red noise. Kept only so a pre-existing
# ``noise_report.yml`` written by that older ``noise_fit.py`` (whose
# ``param_names`` are of the form "<psr>_efac", not "<psr>_<backend>_efac")
# still resolves into a fixed noise model here rather than erroring - see
# ``_is_legacy_noise_params``/``_build_joint_pta``. Any *new* Phase-1 run
# produces per-backend-suffixed names instead, so this legacy path is a
# deliberately narrow one-way compatibility shim, not the general case.
FIXED_NOISE_PARAMS = ("efac", "log10_t2equad", "red_noise_gamma", "red_noise_log10_A")


def _is_legacy_noise_params(noise_params):
    """
    True if ``noise_params`` (a pulsar's fixed-noise dict, keyed by
    unprefixed parameter name) matches exactly the old pre-per-backend
    ``FIXED_NOISE_PARAMS`` shape: a single global EFAC/t2equad pair and a
    single achromatic red-noise pair, nothing else. Used by
    ``_build_joint_pta`` to fall back to the old ``selections.no_selection``,
    no-ECORR, no-DM-noise model structure for a noise report generated
    before this module supported per-backend noise, rather than building a
    per-backend model that a legacy dict could never fully populate.
    """
    return set(noise_params) == set(FIXED_NOISE_PARAMS)


@dataclasses.dataclass
class GWBFitReport:
    pulsars: list
    ntoas: dict
    param_names: list
    posterior_means: list
    n_samples: int
    acceptance_fraction: float | None
    sampler: str
    status: str
    notes: list
    # Parameters held fixed (e.g. gwb_gamma at 13/3), 5/50/95% summaries of
    # the sampled ones, and whether the data constrain the amplitude well
    # enough for a downstream search to fix it at the posterior mean.
    fixed: dict | None = None
    summary: dict | None = None
    amplitude_constrained: bool | None = None

    def save(self, path):
        with open(path, "w") as f:
            yaml.safe_dump(dataclasses.asdict(self), f, sort_keys=False)


def _as_tim_list(tim_files):
    return [tim_files] if isinstance(tim_files, (str, Path)) else list(tim_files)


# Prior on the common process's log10 amplitude, and the fraction of its SD
# the posterior SD must be below for the amplitude to count as constrained.
GWB_LOG10_A_PRIOR = (-20.0, -11.0)
CONSTRAINED_SD_FRACTION = 0.5


def _summaries(post_burn, names):
    return {
        name: {k: round(float(v), 4) for k, v in zip(("p05", "p50", "p95"), np.percentile(post_burn[:, i], [5, 50, 95]))}
        | {"sd": round(float(np.std(post_burn[:, i])), 4)}
        for i, name in enumerate(names)
    }


def _amplitude_constrained(post_burn, names):
    if "gwb_log10_A" not in names or not len(post_burn):
        return None
    lo, hi = GWB_LOG10_A_PRIOR
    prior_sd = (hi - lo) / np.sqrt(12.0)
    return bool(np.std(post_burn[:, list(names).index("gwb_log10_A")]) < CONSTRAINED_SD_FRACTION * prior_sd)


def _build_joint_pta(
    pulsars, red_noise_components=10, dm_noise_components=10, gwb_components=10, gwb_gamma=None
):
    """
    Build a real, joint ``enterprise`` PTA likelihood across every pulsar in
    ``pulsars``: each pulsar's full non-timing-model noise model (per-backend
    white noise, optional ECORR, red noise, optional DM noise), all held
    **fixed** at Phase-1's posterior means, plus one common process (the
    Hellings-Downs-correlated GWB / shared cubic-in-time jerk term) sampled
    jointly across the whole array. Each pulsar's timing model is
    marginalised (``gp_signals.TimingModel``), exactly as in Phase 1 - not
    fixed, since a marginalised timing model has no posterior mean to fix at
    in the first place (it's projected out analytically, not sampled).

    Parameters
    ----------
    pulsars : list of dict
        One entry per pulsar, each with keys ``"name"``, ``"par"``, ``"tim"``
        (a path or list of paths), and ``"noise_params"``: a dict of every
        fixed non-timing-model noise parameter for that pulsar, keyed by its
        unprefixed name (e.g. ``"430_ASP_efac"``, ``"red_noise_log10_A"``,
        ``"dm_gp_log10_A"``) - see the module docstring, and
        ``asimov_ptadata.gwb._resolve_subject_data`` for how this is built
        from a Phase-1 ``noise_report.yml``.
    red_noise_components, dm_noise_components : int
        Number of Fourier components for each pulsar's (fixed) red-noise and
        (if present) DM-noise GPs.
    gwb_components : int
        Number of Fourier components for the shared common process.
    gwb_gamma : float, optional
        Hold the common process's spectral index fixed at this value
        (e.g. ``13/3``) instead of sampling it.

    Returns
    -------
    (list of enterprise.pulsar.Pulsar, enterprise.signals.signal_base.PTA)
    """
    from enterprise.pulsar import Pulsar
    from enterprise.signals import gp_priors, gp_signals, parameter, selections, signal_base, utils

    from .noise_fit import _build_noise_model

    # Instantiated exactly once, with an explicit shared name - see the
    # module docstring for why this (rather than calling parameter.Uniform()
    # again per pulsar) is what makes these genuinely shared. Same prior
    # ranges noise_fit.py uses for a single pulsar's own red noise, reused
    # here for consistency rather than picking a different literature
    # convention.
    log10_A_gw = parameter.Uniform(*GWB_LOG10_A_PRIOR)("gwb_log10_A")
    # A fixed spectral index (13/3 for a background from circular,
    # GW-driven supermassive black-hole binaries) when given; sampling it
    # with few pulsars just returns the prior.
    if gwb_gamma is None:
        gamma_gw = parameter.Uniform(0, 7)("gwb_gamma")
    else:
        gamma_gw = parameter.Constant(float(gwb_gamma))("gwb_gamma")
    gwb_prior = gp_priors.powerlaw(log10_A=log10_A_gw, gamma=gamma_gw)

    orf = utils.hd_orf()  # must be instantiated - see module docstring.
    gwb = gp_signals.FourierBasisCommonGP(gwb_prior, orf, components=gwb_components, name="gw")

    psrs = []
    signalcollections = []
    fixed_values = {}
    for entry in pulsars:
        tim_files = _as_tim_list(entry["tim"])
        tim_arg = [str(t) for t in tim_files] if len(tim_files) > 1 else str(tim_files[0])
        psr = Pulsar(str(entry["par"]), tim_arg, timing_package="pint")
        psrs.append(psr)

        noise_params = entry["noise_params"]
        legacy = _is_legacy_noise_params(noise_params)
        # "log10_ecorr" (bare, no leading underscore) matches a pulsar whose
        # data carries no backend-identifying tim-file flags at all: per
        # noise_fit.py's docstring, enterprise.pulsar.BasePulsar.backend_flags
        # then falls back to a single "" (empty-string) backend for every
        # TOA, and selections.by_backend's own naming
        # (enterprise/signals/selections.py's Selection.__call__) omits an
        # empty selection key from the parameter name rather than using it
        # as a literal "" prefix - confirmed directly against a real
        # flag-free .tim file while building this.
        use_ecorr = (not legacy) and any(
            name == "log10_ecorr" or name.endswith("_log10_ecorr") for name in noise_params
        )
        use_dm_noise = (not legacy) and ("dm_gp_log10_A" in noise_params)

        noise_model = _build_noise_model(
            red_noise_components=red_noise_components,
            dm_noise_components=dm_noise_components,
            use_ecorr=use_ecorr,
            use_dm_noise=use_dm_noise,
            fixed=True,
            selection_fn=selections.no_selection if legacy else None,
        )
        model = noise_model + gwb
        signalcollections.append(model(psr))

        fixed_values.update({f"{psr.name}_{name}": value for name, value in noise_params.items()})

    pta = signal_base.PTA(signalcollections)
    pta.set_default_params(fixed_values)
    return psrs, pta


def run_gwb_fit(
    pulsars,
    outdir,
    niter=6000,
    burn=1000,
    cov_update=None,
    red_noise_components=10,
    dm_noise_components=10,
    gwb_components=10,
    seed=None,
    gwb_gamma=None,
):
    """
    Run a real, joint array-wide GWB common-process fit and write the chain
    plus a summary report to ``outdir``.

    Parameters
    ----------
    pulsars : list of dict
        See :func:`_build_joint_pta`.
    outdir : str or Path
        Directory to write ``gwb_report.yml`` and the ``chain/`` directory
        (PTMCMCSampler's own chain files) into.
    niter, burn, cov_update, seed :
        See ``noise_fit.run_noise_fit`` - identical semantics and the same
        ``cov_update``-must-match-``burn`` requirement.
    red_noise_components, dm_noise_components : int
        Number of red-noise / (if present) DM-noise Fourier components per
        pulsar (both fixed processes).
    gwb_components : int
        Number of Fourier components for the shared common process (the only
        sampled signal in this fixed-noise phase).
    gwb_gamma : float, optional
        See :func:`_build_joint_pta`. The report records it under ``fixed``.

    The report's ``amplitude_constrained`` is true when the posterior SD of
    ``gwb_log10_A`` is below half its prior SD: otherwise the posterior
    mean mostly reflects the prior (the 5-pulsar IPTA DR2 pilot gave
    -17.4 +/- 1.6 on a U(-20, -11) prior), and a downstream search should
    not fix the common process at it without a human deciding to.

    Returns
    -------
    GWBFitReport
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if cov_update is None:
        cov_update = burn
    elif cov_update != burn:
        report = GWBFitReport(
            pulsars=[p["name"] for p in pulsars],
            ntoas={},
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
        report.save(outdir / "gwb_report.yml")
        return report

    pulsar_names = [p["name"] for p in pulsars]
    notes = []

    try:
        psrs, pta = _build_joint_pta(
            pulsars,
            red_noise_components=red_noise_components,
            dm_noise_components=dm_noise_components,
            gwb_components=gwb_components,
            gwb_gamma=gwb_gamma,
        )
    except Exception as exc:
        report = GWBFitReport(
            pulsars=pulsar_names,
            ntoas={},
            param_names=[],
            posterior_means=[],
            n_samples=0,
            acceptance_fraction=None,
            sampler="PTMCMCSampler",
            status="failed",
            notes=[f"failed to build joint PTA: {exc}"],
        )
        report.save(outdir / "gwb_report.yml")
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
        # Same reasoning as noise_fit.run_noise_fit's own sampling try/except:
        # Asimov's detect_completion() just polls for gwb_report.yml's
        # existence, so a bare exception here would leave the job polling
        # forever instead of surfacing a visible failure.
        report = GWBFitReport(
            pulsars=pulsar_names,
            ntoas={},
            param_names=[],
            posterior_means=[],
            n_samples=0,
            acceptance_fraction=None,
            sampler="PTMCMCSampler",
            status="failed",
            notes=[f"sampling failed: {exc}"],
        )
        report.save(outdir / "gwb_report.yml")
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

    report = GWBFitReport(
        pulsars=pulsar_names,
        ntoas={psr.name: int(len(psr.toas)) for psr in psrs},
        param_names=list(pta.param_names),
        posterior_means=posterior_means,
        n_samples=int(chain.shape[0]),
        acceptance_fraction=acceptance_fraction,
        sampler="PTMCMCSampler",
        status="complete",
        notes=notes,
        fixed={"gwb_gamma": float(gwb_gamma)} if gwb_gamma is not None else {},
        summary=_summaries(post_burn, list(pta.param_names)),
        amplitude_constrained=_amplitude_constrained(post_burn, list(pta.param_names)),
    )
    if report.amplitude_constrained is False:
        report.notes.append(
            "gwb_log10_A is not constrained (posterior SD above half the prior's): its posterior mean mostly "
            "reflects the prior"
        )
    report.save(outdir / "gwb_report.yml")
    return report
