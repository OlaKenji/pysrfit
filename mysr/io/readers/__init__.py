"""Built-in readers. Add future TRIUMF, ISIS and J-PARC readers here."""

from .musrroot import MusrRootLoader
from .nexus import NexusLoader
from .text import TextSpectrumLoader

# Specific structured formats first, generic text last.
BUILTIN_LOADERS = (MusrRootLoader, NexusLoader, TextSpectrumLoader)

__all__ = ["BUILTIN_LOADERS", "MusrRootLoader", "NexusLoader", "TextSpectrumLoader"]

