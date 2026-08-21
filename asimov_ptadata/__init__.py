"""Pulsar timing array data acquisition and reduction, with an Asimov pipeline plugin."""

__all__ = ["Pipeline"]


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
        except ImportError as exc:
            # asimov isn't installed - the ptadata CLI (fetch/reduce) still
            # works standalone, it just doesn't expose the asimov plugin
            # class. Match normal attribute-access failure semantics rather
            # than propagating the ImportError.
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
