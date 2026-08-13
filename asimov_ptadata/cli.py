"""Command-line interface for ptadata: fetch and reduce pulsar timing array data."""

from pathlib import Path

import click
import yaml

from . import fetch as fetch_
from . import reduce as reduce_
from . import sources


def _echo_paths(manifest):
    plain = {}
    for key, value in manifest.items():
        plain[key] = [str(v) for v in value] if isinstance(value, list) else str(value)
    click.echo(yaml.safe_dump(plain, sort_keys=False))


def _resolve_release_root(release_root, release_name, cache_dir):
    if bool(release_root) == bool(release_name):
        raise click.UsageError("Pass exactly one of --release or --release-name")
    if release_name:
        return sources.resolve_release(release_name, cache_dir)
    return release_root


@click.group()
def main():
    """ptadata: fetch and reduce pulsar timing array data."""


@main.command()
@click.option("--pulsar", required=True, help="Pulsar name, e.g. J1713+0747")
@click.option(
    "--release", "release_root",
    type=click.Path(exists=True, file_okay=False),
    help="Path to a pinned local data-release checkout",
)
@click.option(
    "--release-name", type=click.Choice(sorted(sources.RELEASES)),
    help="Name of a known release to clone automatically, as an alternative to --release",
)
@click.option(
    "--cache-dir", type=click.Path(file_okay=False), default="./ptadata-cache", show_default=True,
    help="Where cloned releases are cached when using --release-name",
)
@click.option("--outdir", required=True, type=click.Path(file_okay=False), help="Directory to stage files into")
def fetch(pulsar, release_root, release_name, cache_dir, outdir):
    """Locate and stage a pulsar's par/tim/clock files from a data release."""
    release_root = _resolve_release_root(release_root, release_name, cache_dir)
    manifest = fetch_.fetch_pulsar(pulsar, release_root, outdir)
    _echo_paths(manifest)


@main.command()
@click.option("--par", "par_file", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--tim", "tim_files", required=True, multiple=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--outdir", required=True, type=click.Path(file_okay=False))
@click.option("--sigma-threshold", default=5.0, show_default=True)
@click.option("--refit/--no-refit", default=True, show_default=True)
@click.option("--ephem", default=None)
@click.option("--bipm-version", default=None)
def reduce(par_file, tim_files, outdir, sigma_threshold, refit, ephem, bipm_version):
    """Validate and reduce TOAs for a single pulsar, writing cleaned data and a QC report."""
    report = reduce_.reduce_pulsar(
        par_file, list(tim_files), outdir,
        sigma_threshold=sigma_threshold, do_refit=refit,
        ephem=ephem, bipm_version=bipm_version,
    )
    click.echo(yaml.safe_dump(report.__dict__, sort_keys=False))
    if report.status != "pass":
        raise SystemExit(1)


@main.command()
@click.option("--settings", "settings_file", required=True, type=click.Path(exists=True, dir_okay=False))
def run(settings_file):
    """Fetch and reduce in one step, driven by a settings file (used by the Asimov plugin)."""
    with open(settings_file) as f:
        settings = yaml.safe_load(f)

    pulsar = settings["pulsar"]
    rundir = Path(settings["rundir"])
    reduction = settings.get("reduction", {})
    release = settings["release"]

    if "name" in release:
        release_root = sources.resolve_release(release["name"], rundir / "release-cache")
    else:
        release_root = release["root"]

    staged = fetch_.fetch_pulsar(pulsar, release_root, rundir / "staged")
    report = reduce_.reduce_pulsar(
        staged["par"], staged["tim"], rundir / "reduced",
        sigma_threshold=reduction.get("sigma threshold", 5.0),
        do_refit=reduction.get("refit", True),
        ephem=settings.get("ephemeris"),
        bipm_version=settings.get("bipm version"),
    )
    click.echo(f"ptadata run complete for {pulsar}: status={report.status}")
    if report.status != "pass":
        raise SystemExit(1)


@main.command("list-releases")
def list_releases():
    """List known PTA data releases that can be fetched by name."""
    for name, spec in sorted(sources.RELEASES.items()):
        click.echo(f"{name}: {spec['url']} (ref: {spec['ref']})")
