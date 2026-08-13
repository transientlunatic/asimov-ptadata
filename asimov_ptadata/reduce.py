"""PINT-based reduction and quality control for pulsar timing data."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import yaml

import pint.fitter
import pint.logging
import pint.models
import pint.residuals
import pint.toa


@dataclasses.dataclass
class QualityReport:
    pulsar: str
    ntoas: int
    ntoas_flagged: int
    residual_rms_us: float
    residual_rms_us_refit: float | None
    ephem: str | None
    clock: str | None
    units: str | None
    backends: list
    warnings: list
    status: str
    notes: list

    def save(self, path):
        with open(path, "w") as f:
            yaml.safe_dump(dataclasses.asdict(self), f, sort_keys=False)


def _param_value(model, name):
    param = getattr(model, name, None)
    return param.value if param is not None else None


def load_toas(par_file, tim_files, ephem=None, bipm_version=None, planets=True):
    """
    Load a timing model and TOAs, capturing WARNING+ messages PINT raises
    (e.g. a T2-to-ELL1 binary model conversion, stale clock files, ephemeris
    fallbacks) via a loguru sink installed before the model is even loaded -
    checked against a real IPTA DR2 pulsar, where the binary-model
    conversion warning fires during get_model(), before any TOA is read.

    PINT's own diagnostics go through loguru, not the stdlib `warnings`
    module, and its clock-correction messages are routine at INFO level, so
    only WARNING-and-above is a usable "something needs review" signal.

    allow_T2 is on because real release par files commonly carry the generic
    tempo2 "BINARY T2" model, which PINT only accepts if told to convert it
    (to ELL1 here) - refusing it outright would reject a large fraction of
    real data. The conversion itself is exactly the kind of thing the
    WARNING capture above is meant to flag for review.
    """
    records = []
    pint.logging.setup(level="WARNING", sink=lambda m: records.append(m.record["message"]))

    model = pint.models.get_model(str(par_file), allow_T2=True)

    kwargs = {"planets": planets}
    if ephem:
        kwargs["ephem"] = ephem
    if bipm_version:
        kwargs["bipm_version"] = bipm_version

    tim_arg = [str(t) for t in tim_files] if len(tim_files) > 1 else str(tim_files[0])
    toas = pint.toa.get_TOAs(tim_arg, model=model, **kwargs)

    return model, toas, records


def backend_inventory(toas):
    try:
        values, _ = toas.get_flag_value("be")
        return sorted({v for v in values if v is not None})
    except Exception:
        return []


def flag_outliers(toas, model, sigma_threshold=5.0):
    """
    Return (keep_mask, residuals): a robust-sigma mask against the median
    residual, using MAD rather than std so a handful of genuine outliers
    don't inflate the threshold that's meant to catch them.
    """
    residuals = pint.residuals.Residuals(toas, model)
    resids_us = residuals.time_resids.to_value("us")
    median = np.median(resids_us)
    mad = np.median(np.abs(resids_us - median)) * 1.4826
    scale = mad if mad > 0 else np.std(resids_us)
    keep = np.abs(resids_us - median) < sigma_threshold * scale
    return keep, residuals


def refit(model, toas):
    """Run a basic weighted least-squares fit, returning the fitter and its post-fit residual RMS in microseconds."""
    fitter = pint.fitter.WLSFitter(toas, model)
    fitter.fit_toas()
    rms_us = float(fitter.resids.time_resids.std().to_value("us"))
    return fitter, rms_us


def reduce_pulsar(par_file, tim_files, outdir, sigma_threshold=5.0, do_refit=True, ephem=None, bipm_version=None):
    """Load, validate, and reduce a pulsar's timing data, writing cleaned outputs and a QC report to outdir."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        model, toas, warning_records = load_toas(par_file, tim_files, ephem=ephem, bipm_version=bipm_version)
    except Exception as exc:
        report = QualityReport(
            pulsar=Path(par_file).stem,
            ntoas=0,
            ntoas_flagged=0,
            residual_rms_us=float("nan"),
            residual_rms_us_refit=None,
            ephem=None,
            clock=None,
            units=None,
            backends=[],
            warnings=[],
            status="failed",
            notes=[f"failed to load model/TOAs: {exc}"],
        )
        report.save(outdir / "qc_report.yml")
        return report

    keep, residuals = flag_outliers(toas, model, sigma_threshold=sigma_threshold)

    psr_name = model.PSR.value
    rms_us = float(residuals.time_resids.std().to_value("us"))

    good_toas = toas[keep]
    bad_toas = toas[~keep]

    notes = []
    rms_us_refit = None
    if do_refit:
        try:
            _, rms_us_refit = refit(model, good_toas)
        except Exception as exc:
            notes.append(f"refit failed: {exc}")

    good_path = outdir / f"{psr_name}.tim"
    good_toas.write_TOA_file(str(good_path), format="tempo2")

    if bad_toas.ntoas:
        quarantine_path = outdir / f"{psr_name}.quarantine.tim"
        bad_toas.write_TOA_file(str(quarantine_path), format="tempo2")

    (outdir / f"{psr_name}.par").write_text(model.as_parfile())

    status = "pass"
    if warning_records:
        status = "needs-review"
        notes.append(f"{len(warning_records)} warning(s) raised while loading TOAs")
    if toas.ntoas and bad_toas.ntoas > 0.05 * toas.ntoas:
        status = "needs-review"
        notes.append(f"{bad_toas.ntoas}/{toas.ntoas} TOAs flagged as outliers (>5%)")

    report = QualityReport(
        pulsar=psr_name,
        ntoas=toas.ntoas,
        ntoas_flagged=int(bad_toas.ntoas),
        residual_rms_us=rms_us,
        residual_rms_us_refit=rms_us_refit,
        ephem=_param_value(model, "EPHEM"),
        clock=_param_value(model, "CLOCK"),
        units=_param_value(model, "UNITS"),
        backends=backend_inventory(toas),
        warnings=warning_records,
        status=status,
        notes=notes,
    )
    report.save(outdir / "qc_report.yml")
    return report
