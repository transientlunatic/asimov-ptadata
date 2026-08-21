"""Asimov ProjectAnalysis pipeline plugin for single-pulsar noise fitting.

Phase 1 of a larger noise/gravitational-wave-background analysis plan (see
the PR description for the full roadmap): this proves that a second, real
analysis stage - a single-pulsar Bayesian noise fit using ``enterprise`` +
PINT + ``PTMCMCSampler`` (the real science logic lives in
``asimov_ptadata.noise_fit``) - can be driven through Asimov as a
``ProjectAnalysis``, gated on the existing ``ptadata reduce`` stage's QC
result via Asimov's real review/dependency machinery
(``asimov.review``, ``Analysis.matches_filter``,
``ProjectAnalysis.resolve_analyses``), and actually run to completion on
real HTCondor.

Deliberately out of scope for this phase: any cross-pulsar common process /
gravitational-wave-background signal. Each subject gets its own independent
single-pulsar noise fit; a future phase adds the array-wide search on top of
this same plumbing.

A blueprint for this pipeline looks like::

    kind: projectanalysis
    name: noise-fit
    pipeline: ptadata-noise
    subjects:
      - <pulsar name>
    analyses:
      - - "pipeline: ptadata"
        - "review: approved"

The nested list under ``analyses`` is asimov's AND-group smart-dependency
syntax (see ``docs/source/analyses.rst`` in asimov core): it resolves, per
subject, only the ``ptadata`` reduce production(s) whose
``review.status == "APPROVED"`` (asimov's own ``Review``/``ReviewMessage``
states are uppercase; the blueprint's ``"review: approved"`` string matches
case-insensitively via ``Analysis.matches_filter``) - i.e. exactly the
productions ``asimov_ptadata.pipeline.Pipeline.after_completion``
auto-approves when the QC report says ``status: pass``. A single-key dict like
``{pipeline: ptadata, review: approved}`` is *not* equivalent here:
``Analysis._parse_single_dependency`` only accepts single-key dicts (or an
``optional: true`` pair), so combining two conditions requires the AND-group
form above.
"""

import importlib.resources
import os

import yaml

import asimov.pipeline
from asimov import config
from asimov.pipeline import PipelineException
from asimov.scheduler import JobDescription


class NoisePipeline(asimov.pipeline.Pipeline):
    """
    An asimov ProjectAnalysis pipeline for single-pulsar noise fitting.
    """

    name = "ptadata-noise"
    with importlib.resources.path("asimov_ptadata", "noise_settings_template.yml") as _template_file:
        config_template = _template_file
    _pipeline_command = "ptadata"

    def _resolve_subject_assets(self):
        """
        For each subject in this project analysis, find the review-approved
        ``ptadata`` reduce production and its collected par/tim assets.

        The blueprint's ``analyses:`` smart dependency (see the module
        docstring) is what actually *selects* which reduce productions show
        up in ``self.production.analyses`` in the first place - that's the
        real gating mechanism this phase exists to prove out. This method
        adds a second, defensive check on top of it: nothing stops a human
        (or a stale ledger) from marking this analysis "ready" before that
        selection has anything to offer for a given subject (e.g. the
        reduce job for one pulsar in a multi-pulsar ``subjects:`` list
        hasn't finished, or failed QC review). Raising here makes that a
        loud, actionable failure rather than a job silently submitted with
        missing data.

        Returns
        -------
        dict
            Mapping of subject name -> ``{"par": path, "tim": [paths]}``.

        Raises
        ------
        asimov.pipeline.PipelineException
            If any subject has no review-approved ``ptadata`` reduce
            production (with collected par/tim assets) available yet.
        """
        approved_by_subject = {}
        for analysis in self.production.analyses:
            if getattr(analysis.pipeline, "name", None) != "ptadata":
                continue
            if analysis.review.status != "APPROVED":
                continue
            # If more than one review-approved reduce production exists for
            # the same subject, this just keeps the last one encountered in
            # self.production.analyses's iteration order - not necessarily
            # the most recently reviewed one. Deliberately not resolved here:
            # flagged as an open question in the PR for whoever decides the
            # policy (most-recent-production vs. most-recent-review-message
            # both need a real ordering source, and it isn't obvious which
            # one is "right" without knowing how this will actually be used).
            approved_by_subject[analysis.event.name] = analysis

        resolved = {}
        missing = []
        for subject in self.production.subjects:
            analysis = approved_by_subject.get(subject.name)
            assets = analysis.pipeline.collect_assets() if analysis else {}
            if "par" not in assets or "tim" not in assets:
                missing.append(subject.name)
                continue
            resolved[subject.name] = {"par": assets["par"], "tim": assets["tim"]}

        if missing:
            raise PipelineException(
                "No review-approved ptadata reduce production (with "
                "collected par/tim assets) was found for subject(s): "
                f"{', '.join(missing)}. Run `asimov review add <event> "
                "<production> approved` to approve the reduce production, "
                "or wait for automated QC to do so, before submitting this "
                "noise-fit analysis.",
                production=self.production,
            )
        return resolved

    def build_dag(self, dryrun=False):
        name = self.production.name
        os.makedirs(self.production.rundir, exist_ok=True)

        subject_assets = self._resolve_subject_assets()

        sampler_meta = self.production.meta.get("sampler", {}) or {}
        settings = {
            "analysis": name,
            "rundir": self.production.rundir,
            "subjects": [
                {"name": subject_name, "par": assets["par"], "tim": assets["tim"]}
                for subject_name, assets in subject_assets.items()
            ],
            "sampler": {
                "niter": sampler_meta.get("niter", 6000),
                "burn": sampler_meta.get("burn", 1000),
                "red noise components": sampler_meta.get("red noise components", 10),
            },
        }
        settings_file = os.path.join(self.production.rundir, f"{name}.settings.yml")
        with open(settings_file, "w") as f:
            yaml.safe_dump(settings, f, sort_keys=False)

        executable = os.path.join(
            config.get("pipelines", "environment"), "bin", self._pipeline_command
        )

        job = JobDescription(
            executable=executable,
            output=os.path.join(self.production.rundir, f"{name}.out"),
            error=os.path.join(self.production.rundir, f"{name}.err"),
            log=os.path.join(self.production.rundir, f"{name}.log"),
            arguments=f"noise-run --settings {settings_file}",
            memory=self.production.meta.get("scheduler", {}).get("request memory", "2GB"),
            disk=self.production.meta.get("scheduler", {}).get("request disk", "2GB"),
            batch_name=f"ptadata-noise/{name}",
        )

        accounting_group = self.production.meta.get("scheduler", {}).get("accounting group", None)
        if accounting_group:
            job.kwargs["accounting_group_user"] = config.get("condor", "user")
            job.kwargs["accounting_group"] = accounting_group
        else:
            self.logger.warning(
                "This job does not supply any accounting information, which may prevent it running on some clusters."
            )

        if dryrun:
            self.logger.info(f"Dry run: would submit {executable} {job.kwargs['arguments']}")
            return

        cluster_id = self.scheduler.submit(job)
        self.production.job_id = int(cluster_id)
        self._cluster_id = cluster_id
        self.logger.info(f"Submitted {cluster_id} to the job queue.")

    def submit_dag(self, dryrun=False):
        self.build_dag(dryrun=dryrun)
        if not dryrun:
            self.production.status = "running"
        return getattr(self, "_cluster_id", None)

    def _subject_names(self):
        return [subject.name for subject in self.production.subjects]

    def _noise_report_path(self, subject_name):
        return os.path.join(self.production.rundir, "noise", subject_name, "noise_report.yml")

    def detect_completion(self):
        self.logger.info("Checking for completion.")
        subject_names = self._subject_names()
        if subject_names and all(os.path.exists(self._noise_report_path(s)) for s in subject_names):
            self.logger.info("Noise report(s) found, job complete.")
            return True
        self.logger.info("ptadata-noise job completion was not detected.")
        return False

    def after_completion(self):
        self.production.status = "uploaded"

    def collect_assets(self):
        """
        Collect the assets for this job: each subject's noise report and
        raw PTMCMCSampler chain file.
        """
        outputs = {}
        for subject_name in self._subject_names():
            report_path = self._noise_report_path(subject_name)
            if not os.path.exists(report_path):
                continue
            with open(report_path) as f:
                report = yaml.safe_load(f)
            outputs.setdefault("noise reports", {})[subject_name] = report

            chain_path = os.path.join(
                self.production.rundir, "noise", subject_name, "chain", "chain_1.txt"
            )
            if os.path.exists(chain_path):
                outputs.setdefault("chains", {})[subject_name] = chain_path

        return outputs
