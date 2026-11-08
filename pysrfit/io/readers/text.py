"""Reader for already reduced ASCII asymmetry spectra."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..base import Spectrum, UnsupportedFormatError


class TextSpectrumLoader:
    name = "ASCII asymmetry"

    @staticmethod
    def can_load(path: Path) -> bool:
        return path.suffix.lower() in {".txt", ".csv", ".dat", ".xy", ".tsv"}

    @staticmethod
    def load(path: Path) -> Spectrum:
        with path.open(encoding="utf-8-sig") as stream:
            header = "".join(stream.readline() for _ in range(4))
        first_data_line = next(
            (line for line in header.splitlines() if line and not line.lstrip().startswith("#")), "")
        delimiter = "," if "," in first_data_line else None
        array = np.loadtxt(path, comments="#", delimiter=delimiter, ndmin=2)
        if array.ndim != 2 or array.shape[1] not in (2, 3):
            raise UnsupportedFormatError(
                "ASCII spectrum requires 2 or 3 columns: time_us, asymmetry[, sigma]")
        if array.shape[1] == 2:
            raise UnsupportedFormatError(
                "2-column asymmetry has no uncertainties. Add a third sigma column before fitting.")
        time, asymmetry, error = array.T
        if np.any(np.diff(time) <= 0):
            raise ValueError("Time must increase monotonically")
        return Spectrum(time, asymmetry, error, {"source": str(path), "reader": TextSpectrumLoader.name})

