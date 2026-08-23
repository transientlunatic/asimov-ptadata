"""Asimov ProjectAnalysis pipeline plugin for the fixed-noise, array-wide
gravitational-wave-background (GWB) common-process search.

Phase 2 of the larger noise/GWB analysis plan (see ``asimov_ptadata.noise``'s
module docstring for Phase 1): every pulsar's white/red noise parameters are
held fixed at their Phase-1 (``ptadata-noise``) single-pulsar posterior
means, and only the Hellings-Downs-correlated common red-noise process
shared across the whole array is sampled, in one real ``enterprise`` PTA
object built jointly from every subject (the real science logic lives in
``asimov_ptadata.gwb_fit``). Full hierarchical re-fitting of per-pulsar noise
together with the GWB is deliberately out of scope for this phase.

A blueprint for this pipeline looks like::

    kind: projectanalysis
    name: gwb-search
    pipeline: ptadata-gwb
    subjects:
      - <pulsar 1>
      - <pulsar 2>
    analyses:
      - - "pipeline: ptadata-noise"
        - "review: approved"

The ``analyses:`` field above is written in the same AND-group
smart-dependency form ``asimov_ptadata.noise``'s own blueprint uses, and is
kept here purely as human-readable documentation of the intended dependency
- it does **not** drive this pipeline's actual gating, for a reason worth
being explicit about:

Why this pipeline can't just reuse Phase 1's ``analyses:`` mechanism
-----------------------------------------------------------------------
``ptadata-noise`` productions are themselves ``asimov.analysis.ProjectAnalysis``
instances (Phase 1 declares them with ``kind: projectanalysis``), so they are
stored in ``ledger.data["project analyses"]`` (see
``asimov.ledger.Ledger.add_analysis``) rather than being attached to any
subject event's ``.productions``. But
``asimov.analysis.ProjectAnalysis.resolve_analyses()`` (the machinery that
turns a blueprint's ``analyses:`` smart dependency into
``self.production.analyses``) only ever filters each subject event's
``.analyses`` property, and ``asimov.event.Event.analyses`` is simply
``self.productions`` (see ``asimov/event.py``) - it has no knowledge of
project analyses at all.

This was confirmed directly rather than assumed: building two real
``ProjectAnalysis`` objects in a real ledger (one ``ptadata-noise``-pipelined
and review-approved, one with an ``analyses: [["pipeline: ptadata-noise",
"review: approved"]]`` spec depending on it) and calling
``resolve_analyses()`` on the second yields ``self.analyses == []`` every
time - the dependency is structurally invisible to that mechanism, no matter
how the blueprint is phrased. This is a one-hop limitation specific to
depending on *another* ``ProjectAnalysis``; Phase 1's own
``analyses: [["pipeline: ptadata", "review: approved"]]`` works precisely
because ``ptadata`` reduce productions *are* plain per-event
``Production``s living in ``event.productions``.

The fix applied here: ``GWBPipeline._resolve_subject_data()`` walks
``self.production.ledger.project_analyses`` directly, filtering for
``pipeline.name == "ptadata-noise"`` and ``review.status == "APPROVED"`` -
the same real review/dependency primitives Phase 1 uses
(``asimov.review``, ``Analysis.review``), just addressed through the
ledger's own project-analyses list instead of the (here, empty)
``self.production.analyses``. This doesn't need the standard
``analyses:``-driven gate for asimov's CLI to actually attempt this
pipeline, either: ``asimov manage build``/``submit`` (see
``asimov/cli/manage.py``) decide whether to call
``pipeline.build_dag()`` for a project analysis purely from its
``status``/``needs`` fields, not from whether ``resolve_analyses()`` found
anything - the same defensive-raise-on-missing-data pattern
``NoisePipeline._resolve_subject_assets()`` already relies on for exactly
this reason.
"""

import importlib.resources
import os

import yaml

import asimov.pipeline
from asimov import config
from asimov.pipeline import PipelineException
from asimov.review import ReviewMessage
from asimov.scheduler import JobDescription

from .gwb_fit import FIXED_NOISE_PARAMS


class GWBPipeline(asimov.pipeline.Pipeline):
    """
    An asimov ProjectAnalysis pipeline for the fixed-noise array-wide GWB
    common-process search.
    """

    name = "ptadata-gwb"
    with importlib.resources.path("asimov_ptadata", "gwb_settings_template.yml") as _template_file:
        config_template = _template_file
    _pipeline_command = "ptadata"

    def _approved_noise_productions_by_subject(self):
        """
        Map subject (event) name -> the review-approved ``ptadata-noise``
        ``ProjectAnalysis`` covering it, by walking the ledger's project
        analyses directly. See the module docstring for why this - rather
        than ``self.production.analyses`` - is the real dependency source.

        As in ``NoisePipeline._resolve_subject_assets()``: if more than one
        review-approved ``ptadata-noise`` production covers the same
        subject, this keeps the last one encountered in
        ``ledger.project_analyses``'s iteration order - not necessarily the
        most recently approved one. Same open question, deliberately not
        resolved here either.
        """
        approved_by_subject = {}
        for analysis in self.production.ledger.project_analyses:
            if getattr(analysis.pipeline, "name", None) != "ptadata-noise":
                continue
            if analysis.review.status != "APPROVED":
                continue
            for subject in analysis.subjects:
                approved_by_subject[subject.name] = analysis
        return approved_by_subject

    def _resolve_subject_data(self):
        """
        For each subject in this project analysis, find the review-approved
        ``ptadata-noise`` production and pull out what
        ``asimov_ptadata.gwb_fit._build_joint_pta`` needs: par/tim paths and
        the fixed noise parameter values from that pulsar's Phase-1
        posterior means.

        Returns
        -------
        dict
            Mapping of subject name -> the pulsar dict
            ``asimov_ptadata.gwb_fit`` expects (``"name"``, ``"par"``,
            ``"tim"``, plus ``asimov_ptadata.gwb_fit.FIXED_NOISE_PARAMS``).

        Raises
        ------
        asimov.pipeline.PipelineException
            If any subject has no review-approved ``ptadata-noise``
            production with usable par/tim assets and a completed noise
            report available yet.
        """
        approved_by_subject = self._approved_noise_productions_by_subject()

        resolved = {}
        missing = []
        for subject in self.production.subjects:
            analysis = approved_by_subject.get(subject.name)
            if analysis is None:
                missing.append(subject.name)
                continue

            assets = analysis.pipeline.collect_assets()
            par = assets.get("par", {}).get(subject.name)
            tim = assets.get("tim", {}).get(subject.name)
            report = assets.get("noise reports", {}).get(subject.name)
            if not par or not tim or report is None or report.get("status") != "complete":
                missing.append(subject.name)
                continue

            param_names = report.get("param_names", [])
            posterior_means = report.get("posterior_means", [])
            prefix = f"{subject.name}_"
            fixed = {
                name[len(prefix):]: value
                for name, value in zip(param_names, posterior_means)
                if name.startswith(prefix) and name[len(prefix):] in FIXED_NOISE_PARAMS
            }
            if set(FIXED_NOISE_PARAMS) - set(fixed):
                missing.append(subject.name)
                continue

            resolved[subject.name] = {"name": subject.name, "par": par, "tim": tim, **fixed}

        if missing:
            raise PipelineException(
                "No review-approved ptadata-noise production (with a "
                "completed noise report and par/tim assets) was found for "
                f"subject(s): {', '.join(missing)}. Run `asimov review add "
                "<event> <production> approved` to approve the noise-fit "
                "production, or wait for its automated review to do so, "
                "before submitting this GWB-search analysis.",
                production=self.production,
            )
        return resolved

    def build_dag(self, dryrun=False):
        name = self.production.name
        os.makedirs(self.production.rundir, exist_ok=True)

        subject_data = self._resolve_subject_data()

        sampler_meta = self.production.meta.get("sampler", {}) or {}
        settings = {
            "analysis": name,
            "rundir": self.production.rundir,
            # One joint fit across every subject - unlike NoisePipeline's
            # per-subject settings list, this is genuinely a single PTA.
            "pulsars": [subject_data[subject.name] for subject in self.production.subjects],
            "sampler": {
                "niter": sampler_meta.get("niter", 6000),
                "burn": sampler_meta.get("burn", 1000),
                "red noise components": sampler_meta.get("red noise components", 10),
                "gwb components": sampler_meta.get("gwb components", 10),
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
            arguments=f"gwb-run --settings {settings_file}",
            memory=self.production.meta.get("scheduler", {}).get("request memory", "2GB"),
            disk=self.production.meta.get("scheduler", {}).get("request disk", "2GB"),
            batch_name=f"ptadata-gwb/{name}",
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

    def _gwb_dir(self):
        return os.path.join(self.production.rundir, "gwb")

    def _gwb_report_path(self):
        return os.path.join(self._gwb_dir(), "gwb_report.yml")

    def detect_completion(self):
        self.logger.info("Checking for completion.")
        if os.path.exists(self._gwb_report_path()):
            self.logger.info("GWB report found, job complete.")
            return True
        self.logger.info("ptadata-gwb job completion was not detected.")
        return False

    def _read_gwb_report(self):
        report_path = self._gwb_report_path()
        if not os.path.exists(report_path):
            return None
        with open(report_path) as f:
            return yaml.safe_load(f)

    def after_completion(self):
        """
        Runs automatically once the GWB-search job completes, translating
        the report's automated verdict into a real asimov review decision -
        the same pattern ``asimov_ptadata.pipeline.Pipeline`` and
        ``asimov_ptadata.noise.NoisePipeline`` both use, one level further
        up this analysis chain. There is only ever one joint report here
        (unlike ``NoisePipeline``'s per-subject reports), so the mapping is
        direct: ``status: complete`` -> ``APPROVED``, ``status: failed`` ->
        ``REJECTED``, no report yet -> left unreviewed.
        """
        report = self._read_gwb_report()
        if report is not None:
            status = report.get("status", "failed")
            if status == "complete":
                self.production.review.add(
                    ReviewMessage(
                        message="Automated GWB search completed",
                        production=self.production,
                        status="APPROVED",
                    )
                )
            else:
                self.production.review.add(
                    ReviewMessage(
                        message=f"Automated GWB search failed: {report.get('notes')}",
                        production=self.production,
                        status="REJECTED",
                    )
                )

        self.production.status = "uploaded"

    def collect_assets(self):
        """
        Collect the assets for this job: the joint GWB report and raw
        PTMCMCSampler chain file.
        """
        outputs = {}
        report = self._read_gwb_report()
        if report is not None:
            outputs["gwb report"] = report

        chain_path = os.path.join(self._gwb_dir(), "chain", "chain_1.txt")
        if os.path.exists(chain_path):
            outputs["chain"] = chain_path

        return outputs
