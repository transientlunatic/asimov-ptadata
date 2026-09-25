"""Scheduler helpers shared by the Asimov pipeline plugins.

Nothing here imports ``asimov`` at module import time, so it is safe to
import from anywhere in the package.
"""


def job_environment_kwargs(environment, scheduler_type="htcondor"):
    """``JobDescription`` kwargs that set a job's environment variables.

    ``asimov.scheduler.JobDescription`` has no scheduler-agnostic
    environment field, but both backends pass extra keywords through:
    ``to_htcondor`` copies any unmapped kwarg into the submit description,
    so ``environment`` becomes HTCondor's own ``environment`` command (in
    its quoted "new" syntax), and the Slurm scheduler writes each
    ``slurm_<key>`` kwarg as an ``#SBATCH --<key>`` line, so
    ``slurm_export`` becomes ``--export``.

    Parameters
    ----------
    environment : dict or None
        Variable names and values, typically from a production's
        ``scheduler: environment:`` metadata.
    scheduler_type : str
        ``"htcondor"`` or ``"slurm"``.

    Returns
    -------
    dict
        Keyword arguments to merge into ``JobDescription.kwargs``; empty
        when there is nothing to set.
    """
    if not environment:
        return {}
    if scheduler_type == "slurm":
        pairs = []
        for key, value in environment.items():
            if "," in str(value):
                raise ValueError(f"Slurm --export cannot carry a comma in {key}={value!r}")
            pairs.append(f"{key}={value}")
        return {"slurm_export": ",".join(["ALL"] + pairs)}
    return {"environment": '"' + " ".join(_condor_pair(k, v) for k, v in environment.items()) + '"'}


def _condor_pair(key, value):
    """One ``key=value`` in HTCondor's "new" environment syntax.

    Inside the double-quoted string, a value containing whitespace or a
    single quote is wrapped in single quotes with embedded single quotes
    doubled, and every literal double quote is doubled.
    """
    value = str(value)
    if any(c.isspace() for c in value) or "'" in value:
        value = "'" + value.replace("'", "''") + "'"
    return f"{key}={value}".replace('"', '""')


def configured_scheduler_type():
    """The scheduler backend from asimov's configuration, defaulting to HTCondor."""
    from asimov import config

    try:
        return config.get("scheduler", "type")
    except Exception:
        return "htcondor"
