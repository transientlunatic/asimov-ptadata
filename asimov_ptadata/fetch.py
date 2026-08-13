"""Locate and stage pulsar timing data from a pinned data-release checkout.

A PTA data release (IPTA DR, NANOGrav Nyr, EPTA DR, ...) is distributed as a
tagged git repository or archive, not queried live the way GWOSC/CVMFS serve
GW strain data. "Fetching" here means: have a pinned local checkout of a
specific release, then locate and stage one pulsar's files out of it.

Real releases (checked against a live clone of IPTA DR2) commonly lay a
pulsar out as its own directory, with the top-level .tim file pulling in
per-backend files from a "tims/" subdirectory via TEMPO2 INCLUDE lines, and
more than one .par file present for different unit conventions (e.g. a plain
and a ".TDB" variant). That means a pulsar's files generally can't be
flat-copied by glob match alone - the tims/ subdirectory has to move with its
includer file for the relative INCLUDE paths to still resolve, and multiple
.par candidates need to be surfaced rather than guessed at silently, since
picking the wrong unit convention is exactly the kind of thing that silently
breaks array-wide consistency in a combined analysis.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class ReleaseNotFoundError(FileNotFoundError):
    pass


class ReleaseSource:
    """A pinned local checkout of a PTA data release."""

    def __init__(self, release_root):
        self.release_root = Path(release_root)
        if not self.release_root.exists():
            raise ReleaseNotFoundError(f"Release root {self.release_root} does not exist")

    def pulsar_dir(self, psr_name):
        """Return the release's per-pulsar directory for psr_name, or None if this
        release isn't laid out that way (caller should fall back to a flat glob)."""
        matches = [p for p in self.release_root.rglob(psr_name) if p.is_dir()]
        if not matches:
            return None
        if len(matches) > 1:
            raise ReleaseNotFoundError(
                f"Multiple directories named {psr_name} found under {self.release_root} "
                f"(likely parallel data combinations): {matches}. "
                "Point --release at the specific combination you want."
            )
        return matches[0]

    def par_candidates(self, search_dir, psr_name):
        matches = sorted(search_dir.glob(f"{psr_name}*.par"))
        if not matches:
            raise ReleaseNotFoundError(f"No .par file found for {psr_name} under {search_dir}")
        return matches

    def tim_candidates(self, search_dir, psr_name):
        matches = sorted(search_dir.glob(f"{psr_name}*.tim"))
        if not matches:
            raise ReleaseNotFoundError(f"No .tim files found for {psr_name} under {search_dir}")
        return matches

    def clock_dir(self):
        for candidate in ("clock", "clk", "T2runtime/clock"):
            path = self.release_root / candidate
            if path.exists():
                return path
        return None


def _copy_into(src, outdir):
    dst = outdir / src.name
    shutil.copy2(src, dst)
    return dst


def _par_units(path):
    for line in path.read_text().splitlines():
        parts = line.split()
        if parts and parts[0] == "UNITS":
            return parts[1] if len(parts) > 1 else None
    return None


def _select_par(candidates):
    """Prefer a TDB-convention file over filename tidiness.

    PINT only loads UNITS TDB par files (it refuses TCB outright, which is
    tempo2's native convention and common in release par files - checked
    against a real IPTA DR2 pulsar, where the plainest-named file was TCB
    and only the ".TDB"-suffixed variant loaded). Among the remaining
    candidates, prefer the fewest dotted suffixes as a tiebreak.
    """
    non_tcb = [p for p in candidates if _par_units(p) != "TCB"]
    pool = non_tcb if non_tcb else candidates
    return min(pool, key=lambda p: p.stem.count("."))


def clone_release(git_url, tag, dest):
    """Shallow-clone a specific tagged release of a PTA data-release repository."""
    dest = Path(dest)
    if dest.exists():
        return dest
    subprocess.run(
        ["git", "clone", "--branch", tag, "--depth", "1", git_url, str(dest)],
        check=True,
    )
    return dest


def fetch_pulsar(psr_name, release_root, outdir):
    """Stage a pulsar's par/tim/clock files from a release checkout into outdir.

    Returns a manifest dict with keys "par" (the selected file), "par
    candidates" (all matches found), "tim" (list) and, if present, "clock
    dir".
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    source = ReleaseSource(release_root)

    psr_dir = source.pulsar_dir(psr_name)

    if psr_dir is not None:
        # Directory-per-pulsar layout: copy the whole directory so relative
        # INCLUDE paths (e.g. a tims/ subdirectory) keep resolving.
        dest = outdir / psr_name
        shutil.copytree(psr_dir, dest, dirs_exist_ok=True)
        par_candidates = source.par_candidates(dest, psr_name)
        tim_candidates = source.tim_candidates(dest, psr_name)
    else:
        # Flat layout: no per-pulsar directory, so copy the matched files individually.
        par_candidates = [_copy_into(p, outdir) for p in source.par_candidates(source.release_root, psr_name)]
        tim_candidates = [_copy_into(p, outdir) for p in source.tim_candidates(source.release_root, psr_name)]

    manifest = {
        "par": _select_par(par_candidates),
        "par candidates": par_candidates,
        "tim": tim_candidates,
    }

    clock_dir = source.clock_dir()
    if clock_dir is not None:
        manifest["clock dir"] = clock_dir

    return manifest
