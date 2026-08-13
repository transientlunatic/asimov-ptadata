import shutil

import pint.config
import pytest
import yaml
from click.testing import CliRunner

from asimov_ptadata.cli import main


@pytest.fixture
def fake_release(tmp_path):
    psr_dir = tmp_path / "1748-2021E"
    psr_dir.mkdir()
    shutil.copy(pint.config.examplefile("NGC6440E.par"), psr_dir / "1748-2021E.par")
    shutil.copy(pint.config.examplefile("NGC6440E.tim"), psr_dir / "1748-2021E.tim")
    return tmp_path


def test_fetch_command(fake_release, tmp_path):
    runner = CliRunner()
    outdir = tmp_path / "staged"
    result = runner.invoke(
        main, ["fetch", "--pulsar", "1748-2021E", "--release", str(fake_release), "--outdir", str(outdir)]
    )
    assert result.exit_code == 0, result.output
    manifest = yaml.safe_load(result.output)
    assert manifest["par"].endswith("1748-2021E.par")


def test_reduce_command(fake_release, tmp_path):
    runner = CliRunner()
    par = fake_release / "1748-2021E" / "1748-2021E.par"
    tim = fake_release / "1748-2021E" / "1748-2021E.tim"
    outdir = tmp_path / "reduced"
    result = runner.invoke(
        main, ["reduce", "--par", str(par), "--tim", str(tim), "--outdir", str(outdir)]
    )
    assert result.exit_code == 0, result.output
    report = yaml.safe_load(result.output)
    assert report["pulsar"] == "1748-2021E"
    assert (outdir / "qc_report.yml").exists()


def test_run_command(fake_release, tmp_path):
    rundir = tmp_path / "rundir"
    rundir.mkdir()
    settings = {
        "pulsar": "1748-2021E",
        "rundir": str(rundir),
        "release": {"root": str(fake_release)},
        "reduction": {"sigma threshold": 5.0, "refit": True},
    }
    settings_file = rundir / "settings.yml"
    settings_file.write_text(yaml.safe_dump(settings))

    runner = CliRunner()
    result = runner.invoke(main, ["run", "--settings", str(settings_file)])
    assert result.exit_code == 0, result.output
    assert (rundir / "reduced" / "qc_report.yml").exists()
    assert (rundir / "staged" / "1748-2021E" / "1748-2021E.par").exists()
