"""Real-ledger integration tests for asimov_ptadata.gwb.GWBPipeline.

Follows the same real-ledger-integration pattern as ``tests/test_noise.py``:
a real temporary asimov project, a real ``YAMLLedger``, real ``Event``,
``Production`` and ``ProjectAnalysis`` objects, and a mocked scheduler. The
actual enterprise/PINT/PTMCMCSampler joint GWB fit is *not* run here - see
``scripts/gwb_fit_smoketest.py`` for that, for the same reasons
``test_noise.py`` keeps its real fit out of the hermetic pytest suite.

This suite's real coverage target is the two-hop dependency chain (reduce ->
noise -> gwb) - see ``asimov_ptadata.gwb``'s module docstring for why that
chain needs its own resolution logic rather than reusing
``ProjectAnalysis.resolve_analyses()`` the way Phase 1's one-hop dependency
does.
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

from asimov_ptadata.gwb import GWBPipeline

EVENT_BLUEPRINT = """
kind: event
name: {name}
"""

PULSAR_STEMS = {
    "1748-2021E": "NGC6440E",
    "J1028-5819": "J1028-5819-example",
}


class GWBPipelineTests(unittest.TestCase):
    """Tests for asimov_ptadata.gwb.GWBPipeline."""

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

        self.pulsar_names = list(PULSAR_STEMS)
        self.events = {}
        self.release_dir = Path(self.test_dir) / "release"
        for name, stem in PULSAR_STEMS.items():
            blueprint_path = os.path.join(self.test_dir, f"{name}.yaml")
            with open(blueprint_path, "w") as f:
                f.write(EVENT_BLUEPRINT.format(name=name))
            apply_page(file=blueprint_path, event=None, ledger=self.ledger)
            self.events[name] = self.ledger.get_event(name)[0]

            # A fast, network-free par/tim pair per pulsar - PINT's own
            # bundled example data, copied and renamed (two genuinely
            # distinct real pulsars, see PR description for why).
            psr_dir = self.release_dir / name
            psr_dir.mkdir(parents=True)
            shutil.copy(pint.config.examplefile(f"{stem}.par"), psr_dir / f"{name}.par")
            shutil.copy(pint.config.examplefile(f"{stem}.tim"), psr_dir / f"{name}.tim")

    def tearDown(self):
        if self._old_git_config_global is None:
            os.environ.pop("GIT_CONFIG_GLOBAL", None)
        else:
            os.environ["GIT_CONFIG_GLOBAL"] = self._old_git_config_global
        os.chdir(self.cwd)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # -- helpers ----------------------------------------------------------

    def _make_reduce_production(self, subject_name, review_status="pass"):
        event = self.events[subject_name]
        production = Production(
            subject=event,
            name="reduce",
            pipeline="ptadata",
            status="ready",
            ledger=self.ledger,
            rundir=os.path.join(self.test_dir, "run", subject_name, "reduce"),
            data={"release root": str(self.release_dir)},
        )
        event.add_production(production)

        reduced_dir = Path(production.rundir) / "reduced"
        reduced_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(self.release_dir / subject_name / f"{subject_name}.par", reduced_dir / f"{subject_name}.par")
        shutil.copy(self.release_dir / subject_name / f"{subject_name}.tim", reduced_dir / f"{subject_name}.tim")
        report = {"pulsar": subject_name, "ntoas": 10, "ntoas_flagged": 0, "status": review_status}
        (reduced_dir / "qc_report.yml").write_text(yaml.safe_dump(report))
        production.pipeline.after_completion()
        return production

    def _make_noise_analysis(self, subject_name, noise_status="complete", approve=True):
        """
        Build a real ``ptadata-noise`` ProjectAnalysis for a single subject,
        registered in the ledger's project analyses (exactly where
        ``GWBPipeline._resolve_subject_data`` looks), with a real noise
        report and (if ``approve``) a real review approval driven through
        ``NoisePipeline.after_completion`` - not poked directly.
        """
        analysis = ProjectAnalysis(
            name=f"noise-fit-{subject_name}",
            pipeline="ptadata-noise",
            ledger=self.ledger,
            subjects=[subject_name],
            analyses=[["pipeline: ptadata", "review: approved"]],
            status="ready",
        )
        self.ledger.add_analysis(analysis)

        report_dir = Path(analysis.rundir) / "noise" / subject_name
        report_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "pulsar": subject_name,
            "ntoas": 60,
            "param_names": [
                f"{subject_name}_efac",
                f"{subject_name}_log10_t2equad",
                f"{subject_name}_red_noise_gamma",
                f"{subject_name}_red_noise_log10_A",
            ],
            "posterior_means": [1.1, -7.0, 3.5, -14.0],
            "status": noise_status,
        }
        (report_dir / "noise_report.yml").write_text(yaml.safe_dump(report))

        # Drives the review through the real after_completion() flow (same
        # as NoisePipeline's own tests) rather than poking review state
        # directly; `approve=False` leaves the report on disk but skips this,
        # so the analysis stays unreviewed, exactly like a job whose
        # completion hasn't been noticed by `asimov monitor` yet.
        #
        # Crucially, this has to be written back into the ledger's
        # serialised "project analyses" list too
        # (update_analysis_in_project_analysis) - GWBPipeline reads
        # dependencies via self.production.ledger.project_analyses, which
        # *reconstructs* ProjectAnalysis objects from that serialised data
        # (see asimov.ledger.Ledger.project_analyses), not from this live
        # Python object. `asimov monitor` does the same write-back in real
        # use (see asimov/cli/manage.py's submit loop); skipping it here
        # would leave the in-memory review update invisible to
        # GWBPipeline._resolve_subject_data(), exactly as confirmed while
        # writing this test.
        if approve:
            analysis.pipeline.after_completion()
            self.ledger.update_analysis_in_project_analysis(analysis)
        return analysis

    def _make_gwb_analysis(self, name="gwb-search", subjects=None, **meta):
        return ProjectAnalysis(
            name=name,
            pipeline="ptadata-gwb",
            ledger=self.ledger,
            subjects=subjects or self.pulsar_names,
            analyses=[["pipeline: ptadata-noise", "review: approved"]],
            status="ready",
            **meta,
        )

    # -- basic wiring -------------------------------------------------------

    def test_pipeline_class_is_used(self):
        analysis = self._make_gwb_analysis()
        self.assertIsInstance(analysis.pipeline, GWBPipeline)
        self.assertEqual(analysis.pipeline.name, "ptadata-gwb")

    # -- the two-hop dependency resolution (the real point of this phase) --

    def test_standard_analyses_mechanism_cannot_see_project_analyses(self):
        # Documents, with a real assertion, the exact limitation described in
        # asimov_ptadata.gwb's module docstring: resolve_analyses() (the
        # analyses: blueprint mechanism) only scans subject.analyses ==
        # event.productions, which never contains other ProjectAnalysis
        # instances. GWBPipeline therefore cannot rely on
        # self.production.analyses at all, and doesn't.
        for name in self.pulsar_names:
            self._make_reduce_production(name)
            self._make_noise_analysis(name)

        analysis = self._make_gwb_analysis()
        self.assertEqual(analysis.analyses, [])

    def test_resolve_subject_data_builds_pulsars_list_for_all_subjects(self):
        for name in self.pulsar_names:
            self._make_reduce_production(name)
            self._make_noise_analysis(name)

        analysis = self._make_gwb_analysis()
        resolved = analysis.pipeline._resolve_subject_data()

        self.assertEqual(set(resolved), set(self.pulsar_names))
        for name in self.pulsar_names:
            entry = resolved[name]
            self.assertEqual(entry["name"], name)
            self.assertTrue(entry["par"].endswith(f"{name}.par"))
            self.assertTrue(any(t.endswith(f"{name}.tim") for t in entry["tim"]))
            self.assertEqual(entry["efac"], 1.1)
            self.assertEqual(entry["log10_t2equad"], -7.0)
            self.assertEqual(entry["red_noise_gamma"], 3.5)
            self.assertEqual(entry["red_noise_log10_A"], -14.0)

    def test_resolve_subject_data_raises_when_noise_fit_not_approved(self):
        self._make_reduce_production("1748-2021E")
        self._make_noise_analysis("1748-2021E", approve=False)  # never reviewed
        self._make_reduce_production("J1028-5819")
        self._make_noise_analysis("J1028-5819")

        analysis = self._make_gwb_analysis()

        with self.assertRaises(PipelineException) as ctx:
            analysis.pipeline._resolve_subject_data()
        self.assertIn("1748-2021E", str(ctx.exception))

    def test_resolve_subject_data_raises_when_noise_fit_failed(self):
        for name in self.pulsar_names:
            self._make_reduce_production(name)
        self._make_noise_analysis("1748-2021E", noise_status="failed")
        self._make_noise_analysis("J1028-5819")

        analysis = self._make_gwb_analysis()

        with self.assertRaises(PipelineException):
            analysis.pipeline._resolve_subject_data()

    def test_resolve_subject_data_raises_when_no_noise_analysis_exists(self):
        analysis = self._make_gwb_analysis()
        with self.assertRaises(PipelineException):
            analysis.pipeline._resolve_subject_data()

    # -- build_dag / submit_dag against a mocked scheduler ------------------

    def test_build_dag_writes_one_joint_settings_file_and_submits(self):
        for name in self.pulsar_names:
            self._make_reduce_production(name)
            self._make_noise_analysis(name)

        analysis = self._make_gwb_analysis()
        analysis.pipeline._scheduler = MagicMock()
        analysis.pipeline._scheduler.submit = MagicMock(return_value=555)

        analysis.pipeline.build_dag(dryrun=False)

        analysis.pipeline._scheduler.submit.assert_called_once()
        job = analysis.pipeline._scheduler.submit.call_args[0][0]
        self.assertTrue(job.executable.endswith(os.path.join("bin", "ptadata")))
        self.assertIn("gwb-run --settings ", job.kwargs["arguments"])
        self.assertEqual(analysis.job_id, 555)

        settings_file = os.path.join(analysis.rundir, f"{analysis.name}.settings.yml")
        self.assertTrue(os.path.exists(settings_file))
        with open(settings_file) as f:
            settings = yaml.safe_load(f)

        # One joint fit: a single "pulsars" list covering every subject,
        # not a per-subject settings list like NoisePipeline writes.
        self.assertEqual(len(settings["pulsars"]), len(self.pulsar_names))
        self.assertEqual({p["name"] for p in settings["pulsars"]}, set(self.pulsar_names))
        self.assertIn("niter", settings["sampler"])
        self.assertIn("gwb components", settings["sampler"])

    def test_build_dag_raises_when_a_subject_dependency_missing(self):
        self._make_reduce_production("1748-2021E")
        self._make_noise_analysis("1748-2021E")
        # J1028-5819 has no approved noise-fit at all.

        analysis = self._make_gwb_analysis()
        analysis.pipeline._scheduler = MagicMock()

        with self.assertRaises(PipelineException):
            analysis.pipeline.build_dag(dryrun=False)
        analysis.pipeline._scheduler.submit.assert_not_called()

    def test_build_dag_dryrun_does_not_submit(self):
        for name in self.pulsar_names:
            self._make_reduce_production(name)
            self._make_noise_analysis(name)

        analysis = self._make_gwb_analysis()
        analysis.pipeline._scheduler = MagicMock()

        analysis.pipeline.build_dag(dryrun=True)

        analysis.pipeline._scheduler.submit.assert_not_called()
        self.assertIsNone(analysis.job_id)

    def test_submit_dag_sets_status_and_job_id(self):
        for name in self.pulsar_names:
            self._make_reduce_production(name)
            self._make_noise_analysis(name)

        analysis = self._make_gwb_analysis()
        analysis.pipeline._scheduler = MagicMock()
        analysis.pipeline._scheduler.submit = MagicMock(return_value=321)

        job_id = analysis.pipeline.submit_dag(dryrun=False)

        self.assertEqual(job_id, 321)
        self.assertEqual(analysis.status, "running")

    def test_build_dag_supplies_accounting_group_when_configured(self):
        from asimov import config

        for name in self.pulsar_names:
            self._make_reduce_production(name)
            self._make_noise_analysis(name)

        analysis = self._make_gwb_analysis(scheduler={"accounting group": "dept.pta.gwb"})
        analysis.pipeline._scheduler = MagicMock()
        analysis.pipeline._scheduler.submit = MagicMock(return_value=88)

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
        self.assertEqual(job.kwargs["accounting_group"], "dept.pta.gwb")
        self.assertEqual(job.kwargs["accounting_group_user"], "test-user")

    # -- completion / assets -------------------------------------------------

    def test_detect_completion_false_then_true(self):
        for name in self.pulsar_names:
            self._make_reduce_production(name)
            self._make_noise_analysis(name)
        analysis = self._make_gwb_analysis()

        self.assertFalse(analysis.pipeline.detect_completion())

        gwb_dir = Path(analysis.rundir) / "gwb"
        gwb_dir.mkdir(parents=True)
        (gwb_dir / "gwb_report.yml").write_text(yaml.safe_dump({"status": "complete"}))

        self.assertTrue(analysis.pipeline.detect_completion())

    def test_after_completion_approves_review_on_complete(self):
        analysis = self._make_gwb_analysis()
        gwb_dir = Path(analysis.rundir) / "gwb"
        gwb_dir.mkdir(parents=True)
        (gwb_dir / "gwb_report.yml").write_text(yaml.safe_dump({"status": "complete"}))

        analysis.pipeline.after_completion()

        self.assertEqual(analysis.review.status, "APPROVED")
        self.assertEqual(analysis.status, "uploaded")

    def test_after_completion_rejects_review_on_failed(self):
        analysis = self._make_gwb_analysis()
        gwb_dir = Path(analysis.rundir) / "gwb"
        gwb_dir.mkdir(parents=True)
        (gwb_dir / "gwb_report.yml").write_text(yaml.safe_dump({"status": "failed", "notes": ["boom"]}))

        analysis.pipeline.after_completion()

        self.assertEqual(analysis.review.status, "REJECTED")

    def test_after_completion_without_report_leaves_review_untouched(self):
        analysis = self._make_gwb_analysis()
        analysis.pipeline.after_completion()
        self.assertIsNone(analysis.review.status)

    def test_collect_assets_parses_gwb_report_and_chain(self):
        analysis = self._make_gwb_analysis()
        gwb_dir = Path(analysis.rundir) / "gwb"
        gwb_dir.mkdir(parents=True)
        report = {"pulsars": self.pulsar_names, "status": "complete", "posterior_means": [3.5, -14.5]}
        (gwb_dir / "gwb_report.yml").write_text(yaml.safe_dump(report))
        chain_dir = gwb_dir / "chain"
        chain_dir.mkdir()
        (chain_dir / "chain_1.txt").write_text("3.5 -14.5\n")

        assets = analysis.pipeline.collect_assets()

        self.assertEqual(assets["gwb report"]["status"], "complete")
        self.assertTrue(assets["chain"].endswith("chain_1.txt"))

    def test_collect_assets_empty_before_any_report(self):
        analysis = self._make_gwb_analysis()
        self.assertEqual(analysis.pipeline.collect_assets(), {})


if __name__ == "__main__":
    unittest.main()
