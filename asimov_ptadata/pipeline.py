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

    def after_completion(self):
        self.production.status = "uploaded"
        self.production.event.update_data()

    def collect_assets(self):
        """
        Collect the assets for this job.
        """
        outputs = {}
        reduced_dir = os.path.join(self.production.rundir, "reduced")
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

        qc_report_path = os.path.join(reduced_dir, "qc_report.yml")
        if os.path.exists(qc_report_path):
            with open(qc_report_path) as f:
                report = yaml.safe_load(f)
            outputs["qc report"] = report

            data = self.production.event.meta.setdefault("data", {})
            data["ptadata qc"] = report
            self.production.meta["review status"] = report.get("status", "needs-review")

        return outputs
