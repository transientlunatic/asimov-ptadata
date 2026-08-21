"""Asimov pipeline plugin for ptadata.

Submits a single ``ptadata run`` job (fetch + reduce) per pulsar, ahead of
the actual noise/GWB inference pipeline. Mirrors the shape of the gwdata
plugin, but submits through asimov's scheduler-agnostic JobDescription
interface rather than hand-writing HTCondor submit files.
"""

import glob
import importlib.resources
import os

import yaml

import asimov.pipeline
from asimov import config
from asimov.review import ReviewMessage
from asimov.scheduler import JobDescription


class Pipeline(asimov.pipeline.Pipeline):
    """
    An asimov pipeline for ptadata.
    """

    name = "ptadata"
    with importlib.resources.path("asimov_ptadata", "settings_template.yml") as _template_file:
        config_template = _template_file
    _pipeline_command = "ptadata"

    def build_dag(self, dryrun=False):
        name = self.production.name
        os.makedirs(self.production.rundir, exist_ok=True)

        settings_file = self.production.event.repository.find_prods(name, self.category)[0]
        executable = os.path.join(
            config.get("pipelines", "environment"), "bin", self._pipeline_command
        )

        job = JobDescription(
            executable=executable,
            output=os.path.join(self.production.rundir, f"{name}.out"),
            error=os.path.join(self.production.rundir, f"{name}.err"),
            log=os.path.join(self.production.rundir, f"{name}.log"),
            arguments=f"run --settings {settings_file}",
            memory=self.production.meta.get("scheduler", {}).get("request memory", "2GB"),
            disk=self.production.meta.get("scheduler", {}).get("request disk", "2GB"),
            batch_name=f"ptadata/{name}",
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

    def detect_completion(self):
        self.logger.info("Checking for completion.")
        if os.path.exists(os.path.join(self.production.rundir, "reduced", "qc_report.yml")):
            self.logger.info("QC report found, job complete.")
            return True
        self.logger.info("ptadata job completion was not detected.")
        return False

    def _reduced_dir(self):
        return os.path.join(self.production.rundir, "reduced")

    def _read_qc_report(self):
        """
        Read this production's ``qc_report.yml``, if it has been produced yet.

        Returns
        -------
        dict or None
            The parsed QC report, or ``None`` if the reduction hasn't
            written one yet (e.g. the job hasn't completed).
        """
        qc_report_path = os.path.join(self._reduced_dir(), "qc_report.yml")
        if not os.path.exists(qc_report_path):
            return None
        with open(qc_report_path) as f:
            return yaml.safe_load(f)

    def after_completion(self):
        """
        Runs automatically once the reduction job completes (see
        ``asimov.monitor_states``).

        As well as the usual status bookkeeping, this is where the QC
        report's automated verdict is translated into a real asimov review
        decision (``self.production.review``), rather than the ad hoc
        ``production.meta["review status"]`` string this used to be limited
        to. Downstream analyses (e.g. a noise-fit ``ProjectAnalysis``) can
        then gate on this in their blueprint with a ``review: approved``
        smart dependency, using asimov's own
        ``Analysis.matches_filter``/``review.status`` machinery
        (``asimov/review.py``, ``asimov/analysis.py``) instead of a
        pipeline-specific convention:

        - ``qc_report.status == "pass"`` -> an ``APPROVED`` review message is
          added automatically, so the production is immediately usable as a
          dependency.
        - ``"needs-review"`` -> no review message is added. ``review.status``
          is then ``None``, so a ``review: approved`` filter correctly
          excludes it until a human runs ``asimov review add`` to approve or
          reject it.
        - ``"failed"`` -> an explicit ``REJECTED`` review message is added,
          so the failure is visible in the review history rather than the
          production merely being silently excluded from downstream
          dependency resolution.
        """
        report = self._read_qc_report()
        if report is not None:
            status = report.get("status", "needs-review")
            if status == "pass":
                self.production.review.add(
                    ReviewMessage(
                        message="Automated QC passed",
                        production=self.production,
                        status="APPROVED",
                    )
                )
            elif status == "failed":
                self.production.review.add(
                    ReviewMessage(
                        message=(
                            "Automated QC failed: "
                            f"{report.get('ntoas_flagged', '?')}/{report.get('ntoas', '?')} "
                            "TOAs flagged"
                        ),
                        production=self.production,
                        status="REJECTED",
                    )
                )
            # "needs-review": deliberately left unreviewed - see docstring.

        self.production.status = "uploaded"
        self.production.event.update_data()

    def collect_assets(self):
        """
        Collect the assets for this job.
        """
        outputs = {}
        reduced_dir = self._reduced_dir()
        if not os.path.exists(reduced_dir):
            return outputs

        par_files = glob.glob(os.path.join(reduced_dir, "*.par"))
        if par_files:
            outputs["par"] = par_files[0]

        tim_files = [
            f for f in glob.glob(os.path.join(reduced_dir, "*.tim"))
            if not f.endswith(".quarantine.tim")
        ]
        if tim_files:
            outputs["tim"] = tim_files

        quarantine_files = glob.glob(os.path.join(reduced_dir, "*.quarantine.tim"))
        if quarantine_files:
            outputs["quarantined toas"] = quarantine_files

        report = self._read_qc_report()
        if report is not None:
            outputs["qc report"] = report

            data = self.production.event.meta.setdefault("data", {})
            data["ptadata qc"] = report
            # Kept for backwards compatibility with anything already reading
            # this ad hoc key; the real review decision now lives on
            # `self.production.review` (see `after_completion`).
            self.production.meta["review status"] = report.get("status", "needs-review")

        return outputs
