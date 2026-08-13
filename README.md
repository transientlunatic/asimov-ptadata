# asimov-ptadata

Pulsar timing array data acquisition and reduction, usable standalone via the
`ptadata` CLI or as an [Asimov](https://git.ligo.org/asimov/asimov) pipeline
plugin.

`ptadata` locates a pulsar's `.par`/`.tim` files within a pinned checkout of a
PTA data release - either a known release cloned automatically by name, or a
local checkout you already have - then uses
[PINT](https://github.com/nanograv/PINT) to validate and reduce them:
flagging outlier TOAs, checking for clock/ephemeris/binary-model warnings,
optionally running a basic timing-model refit, and writing cleaned
`.par`/`.tim` files plus a machine-readable QC report.

## Standalone use

```console
$ ptadata list-releases
$ ptadata fetch --pulsar J1909-3744 --release-name ipta-dr2 --outdir ./staged
$ ptadata reduce --par ./staged/J1909-3744/J1909-3744.IPTADR2.TDB.par \
    --tim ./staged/J1909-3744/J1909-3744.IPTADR2.tim --outdir ./reduced
```

or against a data release you already have a local checkout of:

```console
$ ptadata fetch --pulsar J1713+0747 --release /path/to/release/checkout --outdir ./staged
```

`ptadata reduce` exits non-zero if the QC report's status isn't `pass`, so it
composes into shell pipelines / CI.

`fetch` copies a pulsar's whole directory when the release lays data out
per-pulsar (checked against a live IPTA DR2 clone: a top-level `.tim` file
commonly `INCLUDE`s per-backend files from a `tims/` subdirectory by relative
path, so that subdirectory has to move with it, not be flattened). When more
than one `.par` candidate is found - real releases often ship a plain file
alongside a unit-conversion variant - the manifest's `"par candidates"` lists
all of them and `"par"` prefers a TDB-convention file, since PINT refuses to
load `UNITS TCB` par files outright (the plainer-looking filename may be the
one that fails).

## As an Asimov pipeline

Installing with the `asimov` extra (`pip install asimov-ptadata[asimov]`)
registers a `ptadata` entry in the `asimov.pipelines` group. A production
using it is driven by a settings file rendered from `production.meta`:

```yaml
kind: analysis
name: reduce
pipeline: ptadata
status: ready
data:
  release name: ipta-dr2   # or: release root: /path/to/release/checkout
reduction:
  sigma threshold: 5.0
  refit: true
```

The plugin submits a single `ptadata run --settings <file>` job through
Asimov's scheduler-agnostic `JobDescription` interface (works with both the
HTCondor and Slurm schedulers), and on completion exposes the reduced
`.par`/`.tim` files and QC report as production assets - `qc_report.yml`'s
`status` field (`pass` / `needs-review`) is written to the production's
`review status` metadata as a review gate ahead of any downstream noise/GWB
analysis.

## Known releases

`ptadata list-releases` prints the current registry (`asimov_ptadata/sources.py`).
IPTA DR2, EPTA DR2, and InPTA DR2 are git repositories and fetched with a
shallow clone. Every URL was checked reachable with `git ls-remote` and the
IPTA DR2 entry checked against a real clone while building this - but
collaborations do restructure/move these between releases, so treat the
registry as a convenience starting point and check the collaboration's own
data page if a lookup starts failing.

**Not yet supported**: NANOGrav's data sets are Zenodo archives, not git
repositories (e.g. the 15yr set, DOI `10.5281/zenodo.16051178`, is a single
~640MB tarball served via the Zenodo REST API), so they need a
download-and-extract path rather than `clone_release`. Not implemented yet.

## Status

Real IPTA DR2 data (`J1909-3744`, 11,483 TOAs across 11 backends) fetches
and reduces successfully end-to-end, including a real T2-to-ELL1 binary
model conversion correctly landing the QC status on `needs-review`. Notably
not yet solved: NANOGrav/Zenodo fetching (above), a `--version`/combination
selector for releases that ship parallel data combinations (IPTA DR2 has
`VersionA`/`VersionB` - point `--release` at the specific one you want), and
pre-staging clock correction files for HPC worker nodes without outbound
internet access (PINT will otherwise try to fetch missing clock files from
the IPTA clock-corrections repository on first use).
