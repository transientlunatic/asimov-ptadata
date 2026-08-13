"""Pulsar timing array data acquisition and reduction, with an Asimov pipeline plugin."""

try:
    from .pipeline import Pipeline
    __all__ = ["Pipeline"]
except ImportError:
    # asimov isn't installed - the ptadata CLI (fetch/reduce) still works
    # standalone, it just doesn't expose the asimov plugin class.
    __all__ = []

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"
