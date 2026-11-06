"""Strict, facility-neutral NeXus NXdata reader."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..base import HistogramRun, Spectrum, UnsupportedFormatError


def _text(value) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if isinstance(value, (list, tuple, np.ndarray)):
        return _text(value[0]) if len(value) else ""
    return str(value)


class NexusLoader:
    name = "NeXus NXdata"

    @staticmethod
    def can_load(path: Path) -> bool:
        if path.suffix.lower() not in {".nxs", ".nexus", ".h5", ".hdf5"}:
            return False
        with path.open("rb") as stream:
            return stream.read(8) == b"\x89HDF\r\n\x1a\n"

    @staticmethod
    def load(path: Path) -> Spectrum | HistogramRun:
        try:
            import h5py
        except (ImportError, ValueError) as exc:
            raise UnsupportedFormatError(
                "NeXus support requires a working 'pip install mysr[nexus]' installation") from exc

        with h5py.File(path, "r") as root:
            groups: list[str] = []

            def collect(_, obj):
                if isinstance(obj, h5py.Group) and _text(obj.attrs.get("NX_class", "")) == "NXdata":
                    groups.append(obj.name)

            root.visititems(collect)
            for name in groups:
                group = root[name]
                signal_name = _text(group.attrs.get("signal", ""))
                if signal_name not in group:
                    continue
                signal = np.asarray(group[signal_name])
                for axis_name in ("time", "time_us", "time_centers", "time_bins"):
                    if axis_name not in group:
                        continue
                    axis_set = group[axis_name]
                    axis = np.asarray(axis_set)
                    if axis.ndim != 1 or signal.shape[-1] != len(axis):
                        continue
                    unit = _text(axis_set.attrs.get("units", "")).lower().strip()
                    scales = {
                        "us": 1.0, "µs": 1.0, "microsecond": 1.0, "microseconds": 1.0,
                        "ns": 0.001, "nanosecond": 0.001, "nanoseconds": 0.001,
                        "ms": 1000.0, "millisecond": 1000.0,
                        "s": 1e6, "second": 1e6,
                    }
                    if unit not in scales:
                        raise UnsupportedFormatError(
                            f"NeXus '{axis_set.name}' time unit '{unit}' is absent or unknown")
                    time = axis.astype(float) * scales[unit]
                    metadata = {
                        "source": str(path), "reader": NexusLoader.name,
                        "nxdata": name, "signal": signal_name,
                    }
                    if signal.ndim == 2:
                        return HistogramRun(time, signal, metadata)
                    if signal.ndim == 1 and "asym" in signal_name.lower():
                        error_name = _text(group.attrs.get("errors", ""))
                        if error_name in group:
                            return Spectrum(time, signal, np.asarray(group[error_name]), metadata)
                        raise UnsupportedFormatError("NeXus asymmetry lacks an error array")
        raise UnsupportedFormatError(
            "No recognized NXdata counts(detector,time) or asymmetry(time) with explicit time units")

