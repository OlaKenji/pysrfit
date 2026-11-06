"""Stable normalized data objects and pluggable raw-data reader interface for mysr."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
import numpy as np


@dataclass(frozen=True)
class DetectorInfo:
    """Facility-independent metadata for one raw histogram detector."""

    number: int
    name: str
    t0_bin: float | None = None
    first_good_bin: int | None = None
    last_good_bin: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self):
        if self.number < 1:
            raise ValueError("Detector numbers are one-based and must be positive")


@dataclass
class Spectrum:
    time_us: np.ndarray
    asymmetry: np.ndarray
    error: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.time_us = np.asarray(self.time_us, dtype=float)
        self.asymmetry = np.asarray(self.asymmetry, dtype=float)
        self.error = np.asarray(self.error, dtype=float)
        if not (self.time_us.ndim == 1 and self.time_us.shape == self.asymmetry.shape == self.error.shape):
            raise ValueError("time, asymmetry and error must be 1D and have the same size")
        if np.any(~np.isfinite(self.time_us)) or np.any(self.error <= 0):
            raise ValueError("Invalid times or nonpositive uncertainties")

    @property
    def values(self) -> np.ndarray:
        return self.asymmetry

    @property
    def kind(self) -> str:
        return "asymmetry"


@dataclass
class CountSpectrum:
    """A packed raw-count spectrum prepared for a musrfit type-0 fit."""

    time_us: np.ndarray
    counts: np.ndarray
    error: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.time_us = np.asarray(self.time_us, dtype=float)
        self.counts = np.asarray(self.counts, dtype=float)
        self.error = np.asarray(self.error, dtype=float)
        if not (self.time_us.ndim == 1 and self.time_us.shape == self.counts.shape == self.error.shape):
            raise ValueError("time, counts and error must be 1D and have the same size")
        if np.any(~np.isfinite(self.time_us)) or np.any(self.counts < 0) or np.any(self.error <= 0):
            raise ValueError("Invalid count spectrum")

    @property
    def values(self) -> np.ndarray:
        return self.counts

    @property
    def kind(self) -> str:
        return "histogram"


@dataclass
class HistogramRun:
    """Counts in shape (detector, time bin), time for unshifted raw channels."""
    time_us: np.ndarray
    counts: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    detectors: list[DetectorInfo] = field(default_factory=list)

    def __post_init__(self):
        self.time_us = np.asarray(self.time_us, dtype=float)
        self.counts = np.asarray(self.counts, dtype=float)
        if self.time_us.ndim != 1 or self.counts.ndim != 2 or self.counts.shape[1] != len(self.time_us):
            raise ValueError("Expected (detectors, timebins) with matching 1D time axis")
        if len(self.time_us) < 2 or np.any(np.diff(self.time_us) <= 0):
            raise ValueError("Time axis must be increasing")
        if np.any(self.counts < 0) or not np.all(np.isfinite(self.counts)):
            raise ValueError("Invalid detector counts")
        if not self.detectors:
            self.detectors = [DetectorInfo(i + 1, f"Detector {i + 1}") for i in range(self.counts.shape[0])]
        if len(self.detectors) != self.counts.shape[0]:
            raise ValueError("Detector metadata count does not match the histogram array")
        numbers = [detector.number for detector in self.detectors]
        if len(set(numbers)) != len(numbers):
            raise ValueError("Detector numbers must be unique")

    @property
    def channel_width_us(self) -> float:
        spacing = np.diff(self.time_us)
        width = float(np.median(spacing))
        if not np.allclose(spacing, width, rtol=1e-6, atol=1e-12):
            raise ValueError("Irregular channel spacing is not supported for histogram alignment")
        return width

    def detector_index(self, reference: int | str) -> int:
        """Resolve a one-based detector number or case-insensitive detector name."""
        if isinstance(reference, str) and not reference.strip().lstrip("+").isdigit():
            wanted = reference.strip().casefold()
            matches = [i for i, detector in enumerate(self.detectors) if detector.name.casefold() == wanted]
        else:
            number = int(reference)
            matches = [i for i, detector in enumerate(self.detectors) if detector.number == number]
        if len(matches) != 1:
            raise ValueError(f"Unknown or ambiguous detector {reference!r}")
        return matches[0]


class DataLoader(Protocol):
    name: str
    def can_load(self, path: Path) -> bool: ...
    def load(self, path: Path) -> Spectrum | HistogramRun: ...


class UnsupportedFormatError(ValueError):
    pass
