"""Fixed-noise, array-wide gravitational-wave-background (GWB) common-process
search: enterprise + PINT + PTMCMCSampler.

Phase 2 walking-skeleton scope, building directly on ``asimov_ptadata.noise_fit``
(Phase 1): every pulsar's white-noise (EFAC + t2equad) and red-noise
(power-law amplitude + spectral index) parameters are held **fixed** at their
Phase-1 single-pulsar posterior means, and only the Hellings-Downs-correlated
common red-noise process shared across the whole array is sampled. Full
hierarchical re-fitting of per-pulsar noise jointly with the GWB is
deliberately out of scope for this phase - see the ``asimov_ptadata.gwb``
module docstring for the roadmap.

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

Fixed noise parameters
------------------------
``enterprise.signals.parameter.Constant(value)`` (not ``Uniform``) is the
correct ``enterprise`` API for a non-sampled fixed parameter - it's a class
factory exactly like ``Uniform``/``Normal``, just with a fixed ``.value``
instead of a prior, so it plugs directly into
``white_signals.MeasurementNoise``/``gp_priors.powerlaw`` the same way
``Uniform(...)`` does in ``noise_fit.py``. Confirmed directly (a
``parameter.Constant``-only PTA builds and evaluates its likelihood/prior
with zero free per-pulsar-noise parameters) while building this.

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

# The fixed per-pulsar noise parameter keys every entry in the ``pulsars``
# list passed to ``run_gwb_fit``/``_build_joint_pta`` must carry, alongside
# "name"/"par"/"tim". These match the *unprefixed* tail of the parameter
# names ``noise_fit.py``'s ``_build_pta`` produces (e.g. a PTA parameter
# named "1748-2021E_red_noise_log10_A" contributes the value keyed here as
# "red_noise_log10_A") - confirmed directly against a real single-pulsar PTA
# while building this. ``asimov_ptadata.gwb._resolve_subject_data`` is what
# actually performs that name-stripping, reading a Phase-1
# ``noise_report.yml``'s ``param_names``/``posterior_means``.
FIXED_NOISE_PARAMS = ("efac", "log10_t2equad", "red_noise_gamma", "red_noise_log10_A")


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

    def save(self, path):
        with open(path, "w") as f:
            yaml.safe_dump(dataclasses.asdict(self), f, sort_keys=False)


def _as_tim_list(tim_files):
    return [tim_files] if isinstance(tim_files, (str, Path)) else list(tim_files)


def _build_joint_pta(pulsars, red_noise_components=10, gwb_components=10):
    """
    Build a real, joint ``enterprise`` PTA likelihood across every pulsar in
    ``pulsars``: per-pulsar EFAC + t2equad white noise and power-law
    red-noise, all held **fixed** at Phase-1's posterior means, plus one
    Hellings-Downs-correlated common red-noise process shared across the
    whole array.

    Parameters
    ----------
    pulsars : list of dict
        One entry per pulsar, each with keys ``"name"``, ``"par"``,
        ``"tim"`` (a path or list of paths), and the fixed noise values
        ``"efac"``, ``"log10_t2equad"``, ``"red_noise_gamma"``,
        ``"red_noise_log10_A"`` (see ``FIXED_NOISE_PARAMS``).
    red_noise_components : int
        Number of Fourier components for each pulsar's (fixed) red-noise GP.
    gwb_components : int
        Number of Fourier components for the shared GWB common process.

    Returns
    -------
    (list of enterprise.pulsar.Pulsar, enterprise.signals.signal_base.PTA)
    """
    from enterprise.pulsar import Pulsar
    from enterprise.signals import gp_priors, gp_signals, parameter, selections, signal_base, utils, white_signals

    selection = selections.Selection(selections.no_selection)

    # Instantiated exactly once, with an explicit shared name - see the
    # module docstring for why this (rather than calling parameter.Uniform()
    # again per pulsar) is what makes these genuinely shared. Same prior
    # ranges noise_fit.py uses for a single pulsar's own red noise, reused
    # here for consistency rather than picking a different literature
    # convention.
    log10_A_gw = parameter.Uniform(-20, -11)("gwb_log10_A")
    gamma_gw = parameter.Uniform(0, 7)("gwb_gamma")
    gwb_prior = gp_priors.powerlaw(log10_A=log10_A_gw, gamma=gamma_gw)

    orf = utils.hd_orf()  # must be instantiated - see module docstring.
    gwb = gp_signals.FourierBasisCommonGP(gwb_prior, orf, components=gwb_components, name="gw")

    psrs = []
    signalcollections = []
    for entry in pulsars:
        tim_files = _as_tim_list(entry["tim"])
        tim_arg = [str(t) for t in tim_files] if len(tim_files) > 1 else str(tim_files[0])
        psr = Pulsar(str(entry["par"]), tim_arg, timing_package="pint")
        psrs.append(psr)

        efac = parameter.Constant(entry["efac"])
        log10_t2equad = parameter.Constant(entry["log10_t2equad"])
        white = white_signals.MeasurementNoise(efac=efac, log10_t2equad=log10_t2equad, selection=selection)

        gamma = parameter.Constant(entry["red_noise_gamma"])
        log10_A = parameter.Constant(entry["red_noise_log10_A"])
        red_prior = gp_priors.powerlaw(log10_A=log10_A, gamma=gamma)
        red_noise = gp_signals.FourierBasisGP(red_prior, components=red_noise_components)

        model = white + red_noise + gwb
        signalcollections.append(model(psr))

    pta = signal_base.PTA(signalcollections)
    return psrs, pta


def run_gwb_fit(
    pulsars,
    outdir,
    niter=6000,
    burn=1000,
    cov_update=None,
    red_noise_components=10,
    gwb_components=10,
    seed=None,
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
    red_noise_components : int
        Number of red-noise Fourier components per pulsar (fixed process).
    gwb_components : int
        Number of Fourier components for the shared GWB process (the only
        sampled signal in this fixed-noise phase).

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
            pulsars, red_noise_components=red_noise_components, gwb_components=gwb_components
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
    )
    report.save(outdir / "gwb_report.yml")
    return report
