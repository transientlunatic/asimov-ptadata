"""A small registry of known PTA data-release repositories.

Every URL here was checked reachable via `git ls-remote` while writing this
(August 2026) and the IPTA DR2 layout was checked against a real clone -
but collaborations do restructure/move these between releases, so treat
this as a convenience starting point, not a permanent index. If a lookup
here starts failing, check the collaboration's own data page rather than
assuming the URL is still current.

Sources:
- IPTA DR2: https://gitlab.com/IPTA/DR2 (see also https://ipta4gw.org/data-release/)
- EPTA DR2: https://gitlab.in2p3.fr/epta/epta-dr2 (also on Zenodo: 10.5281/zenodo.8164424)
- InPTA DR1/DR2: https://github.com/inpta/InPTA.DR1, https://github.com/inpta/InPTA.DR2

NANOGrav's data sets are Zenodo archives rather than git repositories (e.g.
the 15yr set: 10.5281/zenodo.16051178) and aren't handled by this registry
yet - fetching those needs a Zenodo-API download path, not `git clone`.
"""

from pathlib import Path

from .fetch import clone_release

RELEASES = {
    "ipta-dr2": {
        "url": "https://gitlab.com/IPTA/DR2.git",
        "ref": "master",
        "subdir": "release/VersionB",
    },
    "epta-dr2": {
        "url": "https://gitlab.in2p3.fr/epta/epta-dr2.git",
        "ref": "master",
    },
    "inpta-dr2": {
        "url": "https://github.com/inpta/InPTA.DR2.git",
        "ref": "main",
    },
}


def resolve_release(name, cache_dir):
    """Clone (or reuse a cached clone of) a named release, returning its data root."""
    try:
        spec = RELEASES[name]
    except KeyError:
        raise KeyError(f"Unknown release {name!r}, known releases: {sorted(RELEASES)}")

    dest = Path(cache_dir) / name
    clone_release(spec["url"], spec["ref"], dest)
    return dest / spec["subdir"] if "subdir" in spec else dest
