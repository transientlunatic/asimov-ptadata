"""Real-ledger integration tests for asimov_ptadata.noise.NoisePipeline.

Follows the same real-ledger-integration pattern as ``tests/test_pipeline.py``:
a real temporary asimov project, a real ``YAMLLedger``, a real ``Event``, and
real ``Production``/``ProjectAnalysis`` objects constructed directly (rather
than only through blueprint parsing) so their ``.pipeline`` attributes are
real ``asimov_ptadata`` pipeline instances.

The scheduler is mocked (as in test_pipeline.py) and the actual
enterprise/PINT/PTMCMCSampler noise fit is *not* run here - these tests
cover the Asimov plumbing (dependency resolution via asimov's real
``Analysis.matches_filter``/``ProjectAnalysis.resolve_analyses``, and
``build_dag``/``submit_dag``/``detect_completion``/``collect_assets``), not
the science. The real noise fit is exercised separately, standalone
(see ``scripts/noise_fit_smoketest.py``), since it needs a real solar-system
ephemeris (a real network dependency PINT itself has - the same one
``ptadata reduce`` already carries) that would make this suite non-hermetic.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import pint.config
import yaml
from click.testing import CliRunner

from asimov.analysis import ProjectAnalysis
from asimov.cli import project
from asimov.cli.application import apply_page
from asimov.event import Production
from asimov.ledger import YAMLLedger
from asimov.pipeline import PipelineException

from asimov_ptadata.noise import NoisePipeline

EVENT_BLUEPRINT = """
kind: event
name: {name}
"""

# The AND-group smart-dependency form documented in asimov_ptadata.noise's
# module docstring: a single-key dict like {{pipeline: ptadata, review:
# approved}} is *not* equivalent (Analysis._parse_single_dependency only
# accepts one key, or "optional" plus one key), so both conditions have to
# be combined via the nested-list AND-group syntax instead.
ANALYSES_SPEC = [["pipeline: ptadata", "review: approved"]]


class NoisePipelineTests(unittest.TestCase):
    """Tests for asimov_ptadata.noise.NoisePipeline."""

    @classmethod
    def setUpClass(cls):
        cls.cwd = os.getcwd()

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        os.chdir(self.test_dir)

        # See tests/test_pipeline.py for why this is needed: apply_page()'s
        # event repos are `git init` checkouts, and asimov.git.GitRepo hardcodes
        # `git checkout master`.
        git_config_file = Path(self.test_dir) / ".gitconfig-test"
        git_config_file.write_text(
            "[init]\n\tdefaultBranch = master\n"
            "[user]\n\temail = test@example.com\n\tname = Test User\n"
        )
        self._old_git_config_global = os.environ.get("GIT_CONFIG_GLOBAL")
        os.environ["GIT_CONFIG_GLOBAL"] = str(git_config_file)

        runner = CliRunner()
        result = runner.invoke(project.init, ["Test Project", "--root", self.test_dir])
        self.assertEqual(result.exit_code, 0, result.output)

        self.ledger = YAMLLedger(f"{self.test_dir}/.asimov/ledger.yml")

        self.pulsar_name = "1748-2021E"
        blueprint_path = os.path.join(self.test_dir, "event.yaml")
        with open(blueprint_path, "w") as f:
            f.write(EVENT_BLUEPRINT.format(name=self.pulsar_name))
        apply_page(file=blueprint_path, event=None, ledger=self.ledger)
        self.event = self.ledger.get_event(self.pulsar_name)[0]

        # A fast, network-free par/tim pair - PINT's own bundled example
        # data, copied and renamed, as elsewhere in this repo's tests.
        self.release_dir = Path(self.test_dir) / "release"
        psr_dir = self.release_dir / self.pulsar_name
        psr_dir.mkdir(parents=True)
        shutil.copy(pint.config.examplefile("NGC6440E.par"), psr_dir / f"{self.pulsar_name}.par")
        shutil.copy(pint.config.examplefile("NGC6440E.tim"), psr_dir / f"{self.pulsar_name}.tim")

    def tearDown(self):
        if self._old_git_config_global is None:
            os.environ.pop("GIT_CONFIG_GLOBAL", None)
        else:
            os.environ["GIT_CONFIG_GLOBAL"] = self._old_git_config_global
        os.chdir(self.cwd)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # -- helpers ----------------------------------------------------------

    def _make_reduce_production(self, name, review_status=None):
        """
        Build a real `ptadata` reduce Production, register it on the event,
        and (optionally) drive it through the real QC-report -> review flow
        exactly as `asimov_ptadata.pipeline.Pipeline.after_completion` does
        - i.e. via a real `qc_report.yml` and a real `production.review`,
        not by poking review state directly.
        """
        production = Production(
            subject=self.event,
            name=name,
            pipeline="ptadata",
            status="ready",
            ledger=self.ledger,
            rundir=os.path.join(self.test_dir, "run", name),
            data={"release root": str(self.release_dir)},
        )
        self.event.add_production(production)

        if review_status is not None:
            reduced_dir = Path(production.rundir) / "reduced"
            reduced_dir.mkdir(parents=True, exist_ok=True)
            (reduced_dir / f"{self.pulsar_name}.par").write_text("PSR fake\n")
            (reduced_dir / f"{self.pulsar_name}.tim").write_text("FORMAT 1\n")
            report = {
                "pulsar": self.pulsar_name, "ntoas": 10, "ntoas_flagged": 0,
                "status": review_status,
            }
            (reduced_dir / "qc_report.yml").write_text(yaml.safe_dump(report))
            production.pipeline.after_completion()

        return production

    def _make_project_analysis(self, name="noise-fit", subjects=None, analyses=ANALYSES_SPEC, **meta):
        return ProjectAnalysis(
            name=name,
            pipeline="ptadata-noise",
            ledger=self.ledger,
            subjects=subjects or [self.pulsar_name],
            analyses=analyses,
            status="ready",
            **meta,
        )

    # -- basic wiring -------------------------------------------------------

    def test_pipeline_class_is_used(self):
        analysis = self._make_project_analysis()
        self.assertIsInstance(analysis.pipeline, NoisePipeline)
        self.assertEqual(analysis.pipeline.name, "ptadata-noise")

    # -- dependency resolution (the actual review-gating machinery) --------

    def test_resolve_analyses_picks_up_only_approved_reduce_production(self):
        self._make_reduce_production("reduce-pass", review_status="pass")
        self._make_reduce_production("reduce-pending", review_status="needs-review")
        self._make_reduce_production("reduce-failed", review_status="failed")

        analysis = self._make_project_analysis()

        names = {a.name for a in analysis.analyses}
        self.assertEqual(names, {"reduce-pass"})
        self.assertEqual(analysis.analyses[0].review.status, "APPROVED")

    def test_resolve_analyses_empty_when_nothing_approved_yet(self):
        self._make_reduce_production("reduce-pending", review_status="needs-review")

        analysis = self._make_project_analysis()

        self.assertEqual(analysis.analyses, [])

    def test_build_dag_raises_when_dependency_not_approved(self):
        self._make_reduce_production("reduce-pending", review_status="needs-review")
        analysis = self._make_project_analysis()
        analysis.pipeline._scheduler = MagicMock()

        with self.assertRaises(PipelineException):
            analysis.pipeline.build_dag(dryrun=False)

        analysis.pipeline._scheduler.submit.assert_not_called()

    def test_build_dag_raises_when_no_reduce_production_exists_at_all(self):
        analysis = self._make_project_analysis()
        analysis.pipeline._scheduler = MagicMock()

        with self.assertRaises(PipelineException):
            analysis.pipeline.build_dag(dryrun=False)

    # -- build_dag / submit_dag against a mocked scheduler ------------------

    def test_build_dag_writes_settings_and_submits(self):
        self._make_reduce_production("reduce", review_status="pass")
        analysis = self._make_project_analysis()
        analysis.pipeline._scheduler = MagicMock()
        analysis.pipeline._scheduler.submit = MagicMock(return_value=4242)

        analysis.pipeline.build_dag(dryrun=False)

        analysis.pipeline._scheduler.submit.assert_called_once()
        job = analysis.pipeline._scheduler.submit.call_args[0][0]
        self.assertTrue(job.executable.endswith(os.path.join("bin", "ptadata")))
        self.assertIn("noise-run --settings ", job.kwargs["arguments"])
        self.assertEqual(analysis.job_id, 4242)

        settings_file = os.path.join(analysis.rundir, f"{analysis.name}.settings.yml")
        self.assertTrue(os.path.exists(settings_file))
        with open(settings_file) as f:
            settings = yaml.safe_load(f)

        self.assertEqual(len(settings["subjects"]), 1)
        subject_settings = settings["subjects"][0]
        self.assertEqual(subject_settings["name"], self.pulsar_name)
        self.assertTrue(subject_settings["par"].endswith(f"{self.pulsar_name}.par"))
        self.assertEqual(len(subject_settings["tim"]), 1)
        self.assertIn("niter", settings["sampler"])

    def test_build_dag_dryrun_does_not_submit(self):
        self._make_reduce_production("reduce", review_status="pass")
        analysis = self._make_project_analysis()
        analysis.pipeline._scheduler = MagicMock()

        analysis.pipeline.build_dag(dryrun=True)

        analysis.pipeline._scheduler.submit.assert_not_called()
        self.assertIsNone(analysis.job_id)

    def test_submit_dag_sets_status_and_job_id(self):
        self._make_reduce_production("reduce", review_status="pass")
        analysis = self._make_project_analysis()
        analysis.pipeline._scheduler = MagicMock()
        analysis.pipeline._scheduler.submit = MagicMock(return_value=77)

        job_id = analysis.pipeline.submit_dag(dryrun=False)

        self.assertEqual(job_id, 77)
        self.assertEqual(analysis.status, "running")

    def test_build_dag_supplies_accounting_group_when_configured(self):
        from asimov import config

        self._make_reduce_production("reduce", review_status="pass")
        analysis = self._make_project_analysis(scheduler={"accounting group": "dept.pta.noise"})
        analysis.pipeline._scheduler = MagicMock()
        analysis.pipeline._scheduler.submit = MagicMock(return_value=99)

        had_user = config.has_option("condor", "user")
        original_user = config.get("condor", "user") if had_user else None
        config.set("condor", "user", "test-user")
        try:
            analysis.pipeline.build_dag(dryrun=False)
        finally:
            if had_user:
                config.set("condor", "user", original_user)
            else:
                config.remove_option("condor", "user")

        job = analysis.pipeline._scheduler.submit.call_args[0][0]
        self.assertEqual(job.kwargs["accounting_group"], "dept.pta.noise")
        self.assertEqual(job.kwargs["accounting_group_user"], "test-user")

    # -- completion / assets -------------------------------------------------

    def test_detect_completion_false_then_true(self):
        self._make_reduce_production("reduce", review_status="pass")
        analysis = self._make_project_analysis()

        self.assertFalse(analysis.pipeline.detect_completion())

        report_dir = Path(analysis.rundir) / "noise" / self.pulsar_name
        report_dir.mkdir(parents=True)
        (report_dir / "noise_report.yml").write_text(yaml.safe_dump({"status": "complete"}))

        self.assertTrue(analysis.pipeline.detect_completion())

    def test_after_completion_sets_uploaded_status(self):
        analysis = self._make_project_analysis()
        analysis.pipeline.after_completion()
        self.assertEqual(analysis.status, "uploaded")

    def test_collect_assets_parses_noise_reports(self):
        analysis = self._make_project_analysis()

        report_dir = Path(analysis.rundir) / "noise" / self.pulsar_name
        report_dir.mkdir(parents=True)
        report = {
            "pulsar": self.pulsar_name, "ntoas": 62,
            "param_names": ["efac", "gamma"], "posterior_means": [1.1, 2.2],
            "status": "complete",
        }
        (report_dir / "noise_report.yml").write_text(yaml.safe_dump(report))
        chain_dir = report_dir / "chain"
        chain_dir.mkdir()
        (chain_dir / "chain_1.txt").write_text("1.0 2.0\n")

        assets = analysis.pipeline.collect_assets()

        self.assertEqual(assets["noise reports"][self.pulsar_name]["status"], "complete")
        self.assertTrue(assets["chains"][self.pulsar_name].endswith("chain_1.txt"))

    def test_collect_assets_empty_before_any_reports(self):
        analysis = self._make_project_analysis()
        self.assertEqual(analysis.pipeline.collect_assets(), {})

    # -- review-gating (mirrors the reduce Pipeline's after_completion) -----

    def test_collect_assets_exposes_par_and_tim_when_reduce_approved(self):
        # Confirms the GWB pipeline's real dependency surface: par/tim paths
        # exposed directly by NoisePipeline.collect_assets(), not something
        # a caller has to reach two hops back into the reduce production for.
        self._make_reduce_production("reduce", review_status="pass")
        analysis = self._make_project_analysis()

        assets = analysis.pipeline.collect_assets()

        self.assertTrue(assets["par"][self.pulsar_name].endswith(f"{self.pulsar_name}.par"))
        self.assertEqual(len(assets["tim"][self.pulsar_name]), 1)
        self.assertTrue(assets["tim"][self.pulsar_name][0].endswith(f"{self.pulsar_name}.tim"))

    def test_collect_assets_omits_par_and_tim_when_reduce_not_approved(self):
        analysis = self._make_project_analysis()
        assets = analysis.pipeline.collect_assets()
        self.assertNotIn("par", assets)
        self.assertNotIn("tim", assets)

    def _write_noise_report(self, analysis, subject_name, status, **extra):
        report_dir = Path(analysis.rundir) / "noise" / subject_name
        report_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "pulsar": subject_name, "ntoas": 62,
            "param_names": [], "posterior_means": [],
            "status": status,
        }
        report.update(extra)
        (report_dir / "noise_report.yml").write_text(yaml.safe_dump(report))
        return report

    def test_after_completion_approves_review_on_complete(self):
        analysis = self._make_project_analysis()
        self._write_noise_report(analysis, self.pulsar_name, "complete")

        analysis.pipeline.after_completion()

        self.assertEqual(analysis.review.status, "APPROVED")
        self.assertEqual(len(analysis.review), 1)
        self.assertEqual(analysis.status, "uploaded")

    def test_after_completion_rejects_review_on_failed(self):
        analysis = self._make_project_analysis()
        self._write_noise_report(analysis, self.pulsar_name, "failed")

        analysis.pipeline.after_completion()

        self.assertEqual(analysis.review.status, "REJECTED")
        self.assertEqual(len(analysis.review), 1)

    def test_after_completion_rejects_when_any_subject_failed(self):
        for extra_name in ["psrA", "psrB"]:
            bp_path = os.path.join(self.test_dir, f"{extra_name}.yaml")
            with open(bp_path, "w") as f:
                f.write(EVENT_BLUEPRINT.format(name=extra_name))
            apply_page(file=bp_path, event=None, ledger=self.ledger)

        analysis = self._make_project_analysis(subjects=["psrA", "psrB"])
        self._write_noise_report(analysis, "psrA", "complete")
        self._write_noise_report(analysis, "psrB", "failed")

        analysis.pipeline.after_completion()

        self.assertEqual(analysis.review.status, "REJECTED")

    def test_after_completion_without_any_report_leaves_review_untouched(self):
        analysis = self._make_project_analysis()
        analysis.pipeline.after_completion()
        self.assertIsNone(analysis.review.status)

    def test_review_approved_filter_matches_only_approved_analysis(self):
        # Exercises the real asimov filtering machinery a downstream GWB
        # search stage relies on (see asimov_ptadata.gwb's module docstring
        # for why it can't use the standard analyses: mechanism to reach
        # this - it walks ledger.project_analyses directly instead, but
        # still needs a real APPROVED/REJECTED review status here to filter
        # on).
        completed = self._make_project_analysis(name="noise-complete")
        self._write_noise_report(completed, self.pulsar_name, "complete")
        completed.pipeline.after_completion()

        failed = self._make_project_analysis(name="noise-failed")
        self._write_noise_report(failed, self.pulsar_name, "failed")
        failed.pipeline.after_completion()

        self.assertTrue(completed.matches_filter(["review"], "approved"))
        self.assertFalse(failed.matches_filter(["review"], "approved"))


if __name__ == "__main__":
    unittest.main()
