"""Real-ledger integration tests for the asimov_ptadata Asimov pipeline plugin.

Follows the same pattern as asimov core's own
``tests/test_pipelines/test_testing_pipelines.py``: a real temporary asimov
project (``asimov.cli.project.init`` via ``click.testing.CliRunner``), a real
``YAMLLedger``, an event added through ``asimov.cli.application.apply_page``,
and a production (``asimov.event.Production``, i.e.
``GravitationalWaveTransient``) constructed directly so its ``.pipeline``
attribute is a real ``asimov_ptadata.pipeline.Pipeline`` instance.

The scheduler itself is never exercised for real - ``pipeline._scheduler`` is
monkeypatched with a ``MagicMock`` (the pattern asimov core's own tests use),
so these tests need neither a real HTCondor/Slurm scheduler nor network
access. Reduction fixture data is PINT's own bundled ``NGC6440E.par``/``.tim``
example files, copied and renamed - the same trick already used in
``tests/test_cli.py`` - so nothing here depends on a real PTA data release
being reachable.
"""

import logging
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import pint.config
import yaml
from click.testing import CliRunner

from asimov import config
from asimov.cli import project
from asimov.cli.application import apply_page
from asimov.event import Production
from asimov.ledger import YAMLLedger

from asimov_ptadata.pipeline import Pipeline

EVENT_BLUEPRINT = """
kind: event
name: 1748-2021E
"""


class PtadataPipelineTests(unittest.TestCase):
    """Tests for asimov_ptadata.pipeline.Pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.cwd = os.getcwd()

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        os.chdir(self.test_dir)

        runner = CliRunner()
        result = runner.invoke(project.init, ["Test Project", "--root", self.test_dir])
        self.assertEqual(result.exit_code, 0, result.output)

        self.ledger = YAMLLedger(f"{self.test_dir}/.asimov/ledger.yml")

        blueprint_path = os.path.join(self.test_dir, "event.yaml")
        with open(blueprint_path, "w") as f:
            f.write(EVENT_BLUEPRINT)
        apply_page(file=blueprint_path, event=None, ledger=self.ledger)
        self.event = self.ledger.get_event("1748-2021E")[0]

        # A fast, network-free par/tim pair - PINT's own bundled example
        # data, copied and renamed, exactly as tests/test_cli.py does.
        self.release_dir = Path(self.test_dir) / "release"
        psr_dir = self.release_dir / "1748-2021E"
        psr_dir.mkdir(parents=True)
        shutil.copy(pint.config.examplefile("NGC6440E.par"), psr_dir / "1748-2021E.par")
        shutil.copy(pint.config.examplefile("NGC6440E.tim"), psr_dir / "1748-2021E.tim")

    def tearDown(self):
        os.chdir(self.cwd)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _make_production(self, name="reduce", rundir=None, **meta):
        meta.setdefault("data", {"release root": str(self.release_dir)})
        meta.setdefault("reduction", {"sigma threshold": 5.0, "refit": True})
        return Production(
            subject=self.event,
            name=name,
            pipeline="ptadata",
            status="ready",
            ledger=self.ledger,
            rundir=rundir or os.path.join(self.test_dir, "run", name),
            **meta,
        )

    def _settings_file_for(self, production):
        return production.event.repository.find_prods(production.name, production.category)[0]

    def test_pipeline_class_is_used(self):
        production = self._make_production()
        self.assertIsInstance(production.pipeline, Pipeline)
        self.assertEqual(production.pipeline.name, "ptadata")

    def test_build_dag_sets_job_arguments_and_warns_without_accounting_group(self):
        production = self._make_production()
        production.pipeline._scheduler = MagicMock()
        production.pipeline._scheduler.submit = MagicMock(return_value=12345)

        with self.assertLogs("asimov", level="WARNING") as cm:
            production.pipeline.build_dag(dryrun=False)

        self.assertTrue(
            any("accounting information" in message for message in cm.output),
            cm.output,
        )

        production.pipeline._scheduler.submit.assert_called_once()
        job = production.pipeline._scheduler.submit.call_args[0][0]

        settings_file = self._settings_file_for(production)
        self.assertEqual(job.kwargs["arguments"], f"run --settings {settings_file}")
        self.assertTrue(job.executable.endswith(os.path.join("bin", "ptadata")))
        self.assertNotIn("accounting_group", job.kwargs)

        self.assertEqual(production.job_id, 12345)

    def test_build_dag_supplies_accounting_group_when_configured(self):
        production = self._make_production(scheduler={"accounting group": "dept.pta.test"})
        production.pipeline._scheduler = MagicMock()
        production.pipeline._scheduler.submit = MagicMock(return_value=54321)

        had_user = config.has_option("condor", "user")
        original_user = config.get("condor", "user") if had_user else None
        config.set("condor", "user", "test-user")
        try:
            with self.assertNoLogs("asimov", level="WARNING"):
                production.pipeline.build_dag(dryrun=False)
        finally:
            if had_user:
                config.set("condor", "user", original_user)
            else:
                config.remove_option("condor", "user")

        job = production.pipeline._scheduler.submit.call_args[0][0]
        self.assertEqual(job.kwargs["accounting_group"], "dept.pta.test")
        self.assertEqual(job.kwargs["accounting_group_user"], "test-user")
        self.assertEqual(production.job_id, 54321)

    def test_build_dag_dryrun_does_not_submit(self):
        production = self._make_production()
        production.pipeline._scheduler = MagicMock()

        production.pipeline.build_dag(dryrun=True)

        production.pipeline._scheduler.submit.assert_not_called()
        self.assertIsNone(production.job_id)

    def test_submit_dag_sets_status_and_job_id(self):
        production = self._make_production()
        production.pipeline._scheduler = MagicMock()
        production.pipeline._scheduler.submit = MagicMock(return_value=99)

        job_id = production.pipeline.submit_dag(dryrun=False)

        self.assertEqual(job_id, 99)
        self.assertEqual(production.status, "running")
        self.assertEqual(production.job_id, 99)

    def test_detect_completion_false_then_true(self):
        production = self._make_production()
        self.assertFalse(production.pipeline.detect_completion())

        reduced_dir = Path(production.rundir) / "reduced"
        reduced_dir.mkdir(parents=True)
        (reduced_dir / "qc_report.yml").write_text(yaml.safe_dump({"status": "pass"}))

        self.assertTrue(production.pipeline.detect_completion())

    def test_after_completion_sets_uploaded_status(self):
        production = self._make_production()
        production.pipeline.after_completion()
        self.assertEqual(production.status, "uploaded")

    def test_collect_assets_empty_before_rundir_exists(self):
        production = self._make_production()
        self.assertEqual(production.pipeline.collect_assets(), {})

    def test_collect_assets_parses_qc_report_and_sets_review_status(self):
        production = self._make_production()
        reduced_dir = Path(production.rundir) / "reduced"
        reduced_dir.mkdir(parents=True)

        (reduced_dir / "1748-2021E.par").write_text("PSR 1748-2021E\n")
        (reduced_dir / "1748-2021E.tim").write_text("FORMAT 1\n")
        (reduced_dir / "1748-2021E.quarantine.tim").write_text("FORMAT 1\n")

        report = {
            "pulsar": "1748-2021E",
            "ntoas": 10,
            "ntoas_flagged": 1,
            "status": "needs-review",
        }
        (reduced_dir / "qc_report.yml").write_text(yaml.safe_dump(report))

        assets = production.pipeline.collect_assets()

        self.assertTrue(assets["par"].endswith("1748-2021E.par"))
        self.assertEqual(len(assets["tim"]), 1)
        self.assertEqual(len(assets["quarantined toas"]), 1)
        self.assertEqual(assets["qc report"]["status"], "needs-review")

        self.assertEqual(production.meta["review status"], "needs-review")
        self.assertEqual(
            production.event.meta["data"]["ptadata qc"]["status"], "needs-review"
        )


if __name__ == "__main__":
    unittest.main()
