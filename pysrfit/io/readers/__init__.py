"""Built-in raw and reduced-data readers."""

from .musrroot import MusrRootLoader
from .nexus import NexusLoader
from .mud import MudLoader
from .text import TextSpectrumLoader

# Specific structured formats first, generic text last.
BUILTIN_LOADERS = (MusrRootLoader, NexusLoader, MudLoader, TextSpectrumLoader)

__all__ = [
    "BUILTIN_LOADERS", "MusrRootLoader", "NexusLoader", "MudLoader",
    "TextSpectrumLoader",
]

