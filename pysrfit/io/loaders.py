"""Facility-neutral reader registry and data-file auto-detection."""
from __future__ import annotations

from pathlib import Path

from .base import DataLoader, HistogramRun, Spectrum, UnsupportedFormatError


class LoaderRegistry:
    """Ordered registry; specific facility readers should precede generic ones."""

    def __init__(self):
        self._loaders: list[DataLoader] = []

    @property
    def loaders(self) -> tuple[DataLoader, ...]:
        return tuple(self._loaders)

    def register(self, loader: DataLoader) -> DataLoader:
        if any(existing.name == loader.name for existing in self._loaders):
            raise ValueError(f"A data loader named {loader.name!r} is already registered")
        self._loaders.append(loader)
        return loader

    def load(self, path: str | Path) -> Spectrum | HistogramRun:
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Data file not found: {source}")
        for loader in self._loaders:
            if loader.can_load(source):
                return loader.load(source)
        names = ", ".join(loader.name for loader in self._loaders)
        raise UnsupportedFormatError(
            f"No reader recognized {source.name}. Registered readers: {names}")


REGISTRY = LoaderRegistry()
LOADERS = REGISTRY._loaders  # Read-only by convention; prefer register_loader().


def register_loader(loader: DataLoader) -> DataLoader:
    return REGISTRY.register(loader)


def load_data(path: str | Path) -> Spectrum | HistogramRun:
    return REGISTRY.load(path)


# Reader dependencies are imported lazily by each loader so startup remains fast.
from .readers import BUILTIN_LOADERS  # noqa: E402

for _loader in BUILTIN_LOADERS:
    register_loader(_loader)

