"""TRIUMF MUD reader backed by the maintained ``mud-py`` package."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..base import DetectorInfo, HistogramRun, UnsupportedFormatError


_MUD_CORE = b"\x03\x00\x01\x01\x00\x00\x01\x02"


def _optional_bin(value, maximum: int) -> int | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number) or number < 0:
        return None
    return min(int(round(number)), maximum)


def _attribute(obj, name: str, default=""):
    value = getattr(obj, name, default)
    return default if value is None else value


class MudLoader:
    name = "TRIUMF MUD"

    @staticmethod
    def can_load(path: Path) -> bool:
        if path.suffix.lower() not in {".msr", ".mud"}:
            return False
        with path.open("rb") as stream:
            header = stream.read(12)
        # Text musrfit .msr files share the extension but never this binary
        # MUD group/format core.
        return len(header) == 12 and header[4:12] == _MUD_CORE

    @staticmethod
    def load(path: Path) -> HistogramRun:
        try:
            from mudpy import mdata
        except (ImportError, OSError) as exc:
            raise UnsupportedFormatError(
                "TRIUMF MUD support requires a compatible mud-py installation; "
                "reinstall PySRFit with 'python -m pip install --force-reinstall pysrfit'") from exc

        try:
            source = mdata(str(path))
        except Exception as exc:
            raise UnsupportedFormatError(
                f"Cannot read MUD file {path.name}: {exc}") from exc

        facility = str(_attribute(source, "lab")).strip()
        if facility.casefold() != "triumf":
            raise UnsupportedFormatError(
                f"MUD loading is currently restricted to TRIUMF; {facility or path.name!r} "
                "should be imported from its NeXus file instead")

        histograms = sorted(source.hist.values(), key=lambda item: item.id_number)
        if not histograms:
            raise UnsupportedFormatError("TRIUMF MUD file contains no histograms")
        lengths = {len(histogram.data) for histogram in histograms}
        if len(lengths) != 1:
            raise UnsupportedFormatError(
                "TRIUMF MUD histograms have different lengths; ragged runs are unsupported")
        widths_fs = {int(histogram.fs_per_bin) for histogram in histograms}
        if len(widths_fs) != 1 or next(iter(widths_fs)) <= 0:
            raise UnsupportedFormatError(
                "TRIUMF MUD histograms do not share a positive time-bin width")

        length = next(iter(lengths))
        channel_width_us = next(iter(widths_fs)) * 1e-9
        time = np.arange(length, dtype=float) * channel_width_us
        detectors = [
            DetectorInfo(
                number=int(histogram.id_number),
                name=str(histogram.title).strip() or f"Detector {histogram.id_number}",
                t0_bin=_optional_bin(histogram.t0_bin, length - 1),
                first_good_bin=_optional_bin(histogram.good_bin1, length - 1),
                last_good_bin=_optional_bin(histogram.good_bin2, length - 1),
                metadata={
                    "events": int(histogram.n_events),
                    "background_start": _optional_bin(histogram.background1, length - 1),
                    "background_end": _optional_bin(histogram.background2, length - 1),
                },
            )
            for histogram in histograms
        ]
        counts = np.vstack([
            np.asarray(histogram.data, dtype=float) for histogram in histograms])
        metadata = {
            "source": str(path),
            "reader": MudLoader.name,
            "facility": facility,
            "instrument": str(_attribute(source, "area")).strip(),
            "format": "MUD",
            "run_number": int(_attribute(source, "run", 0)),
            "title": str(_attribute(source, "title")).strip(),
            "sample": str(_attribute(source, "sample")).strip(),
            "temperature": str(_attribute(source, "temperature")).strip(),
            "field": str(_attribute(source, "field")).strip(),
            "start_time": str(_attribute(source, "start_date")).strip(),
            "end_time": str(_attribute(source, "end_date")).strip(),
            "experiment_identifier": int(_attribute(source, "exp", 0)),
            "time_resolution_us": channel_width_us,
            "suggested_run_configuration": {},
            "file_header": {
                "method": str(_attribute(source, "method")).strip(),
                "apparatus": str(_attribute(source, "apparatus")).strip(),
                "mode": str(_attribute(source, "mode")).strip(),
                "orientation": str(_attribute(source, "orientation")).strip(),
                "data_acquisition": str(_attribute(source, "das")).strip(),
                "experimenter": str(_attribute(source, "experimenter")).strip(),
                "duration_s": int(_attribute(source, "duration", 0)),
            },
        }
        return HistogramRun(time, counts, metadata, detectors)
