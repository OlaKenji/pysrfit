"""PySRFit: GUI-first muon spin spectroscopy fitting workbench."""

from importlib.metadata import PackageNotFoundError, version

from .app import PySRFitApplication

try:
    __version__ = version("pysrfit")
except PackageNotFoundError:  # Source tree without an installed distribution.
    __version__ = "uninstalled"

__all__ = ["PySRFitApplication", "__version__"]
