import shutil

import pint.config
import yaml
from click.testing import CliRunner

from asimov_ptadata.cli import main
from asimov_ptadata.reduce import load_toas, reduce_pulsar, refit


def _example(tmp_path, extra_par_lines=""):
    par = tmp_path / "1748-2021E.par"
    tim = tmp_path / "1748-2021E.tim"
    par.write_text(open(pint.config.examplefile("NGC6440E.par")).read() + extra_par_lines)
    shutil.copy(pint.config.examplefile("NGC6440E.tim"), tim)
    return par, tim


def test_clipping_disabled_keeps_every_toa(tmp_path):
    par, tim = _example(tmp_path)
    report = reduce_pulsar(par, [tim], tmp_path / "out", sigma_threshold=None, do_refit=False)
    assert report.ntoas_flagged == 0
    assert "outlier clipping disabled" in report.notes
    assert not (tmp_path / "out" / f"{report.pulsar}.quarantine.tim").exists()


def test_clipping_on_by_default_does_not_add_the_note(tmp_path):
    par, tim = _example(tmp_path)
    report = reduce_pulsar(par, [tim], tmp_path / "out", do_refit=False)
    assert "outlier clipping disabled" not in report.notes


def test_refit_freezes_a_jump_that_selects_no_toas(tmp_path):
    # e.g. every TOA of a JUMP clipped as an outlier: PINT refuses to fit an empty mask.
    par, tim = _example(tmp_path, "JUMP -f no_such_backend 0.0 1\n")
    model, toas, _ = load_toas(par, [tim])
    _, rms_us = refit(model, toas)
    assert rms_us > 0
    assert model.JUMP1.frozen


def test_run_honours_clip_outliers_false(tmp_path):
    release = tmp_path / "release"
    (release / "1748-2021E").mkdir(parents=True)
    _example(release / "1748-2021E")
    rundir = tmp_path / "rundir"
    rundir.mkdir()
    settings = {
        "pulsar": "1748-2021E",
        "rundir": str(rundir),
        "release": {"root": str(release)},
        "reduction": {"sigma threshold": 5.0, "refit": True, "clip outliers": False},
    }
    (rundir / "settings.yml").write_text(yaml.safe_dump(settings))

    result = CliRunner().invoke(main, ["run", "--settings", str(rundir / "settings.yml")])

    report = yaml.safe_load((rundir / "reduced" / "qc_report.yml").read_text())
    assert report["ntoas_flagged"] == 0, result.output
    assert "outlier clipping disabled" in report["notes"]
