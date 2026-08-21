"""Pulsar timing array data acquisition and reduction, with an Asimov pipeline plugin."""

import importlib.util

# Only advertise Pipeline when the 'asimov' extra is actually installed -
# otherwise `from asimov_ptadata import *` would try to resolve it via
# __getattr__ below and hit the AttributeError it deliberately raises,
# where previously (before Pipeline's export became lazy) it was just
# absent from __all__. find_spec() only checks whether the top-level
# 'asimov' package is importable; it doesn't import asimov_ptadata.pipeline
# itself, so it can't reintroduce the circular-import this laziness exists
# to avoid.
__all__ = ["Pipeline"] if importlib.util.find_spec("asimov") is not None else []


def __getattr__(name):
    """Lazily import :class:`Pipeline` on first access (PEP 562).

    ``asimov_ptadata.pipeline`` imports ``asimov.pipeline``, which in turn
    imports ``asimov.analysis`` - and *that* module loads every registered
    ``asimov.pipelines`` entry point (including this package's own
    ``ptadata`` entry, ``asimov_ptadata:Pipeline``) as part of building
    asimov's pipeline registry. If ``Pipeline`` were imported eagerly here,
    the *very first* import of this package for any reason (e.g. just using
    the standalone ``ptadata`` CLI, or a test importing
    ``asimov_ptadata.fetch``) would recursively re-enter this
    still-initializing module while asimov's entry-point loader looks it up
    - finding no ``Pipeline`` attribute yet, since we'd still be in the
    middle of importing it. That failure is swallowed by asimov's loader
    (a warning, not an error), so the plugin would then silently be missing
    from ``known_pipelines`` for the rest of the process - reproduced by
    simply reordering imports, e.g. ``import asimov_ptadata.fetch`` before
    anything imports ``asimov.analysis``.

    Deferring the import to attribute-access time (which is exactly when
    asimov's entry-point loader actually asks for ``Pipeline``, well after
    this module itself has finished initialising) breaks that cycle.
    """
    if name == "Pipeline":
        try:
            from .pipeline import Pipeline
        except ModuleNotFoundError as exc:
            # Only treat this as "the optional extra isn't installed" if
            # 'asimov' itself is the missing module - a broad `except
            # ImportError` here would also swallow a real bug inside
            # .pipeline (a missing/renamed sub-dependency, a bad relative
            # import, ...) and misreport it as an absent extra, which is
            # much harder to diagnose than letting it propagate normally.
            if exc.name != "asimov":
                raise
            raise AttributeError(
                "asimov_ptadata.Pipeline requires the 'asimov' extra "
                "(pip install asimov-ptadata[asimov])"
            ) from exc
        return Pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"
