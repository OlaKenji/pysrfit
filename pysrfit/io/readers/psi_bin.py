"""Read the classic PSI-BIN histogram format used by bulk muSR instruments."""
from __future__ import annotations

from pathlib import Path
import re
import struct

import numpy as np

from ..base import DetectorInfo, HistogramRun, UnsupportedFormatError


_FORMAT_ID = b"1N"
_HEADER_SIZE = 1024
_RECORD_SIZE = 512
_MAX_HISTOGRAMS = 16
_INSTRUMENT = re.compile(
    r"(?:^|_)(gps|gpd|ltf|dolly|hifi|hal|flame)(?:_|$)",
    flags=re.IGNORECASE,
)


def _integer(header: bytes, offset: int, size: int = 2) -> int:
    code = "h" if size == 2 else "i"
    return int(struct.unpack_from(f"<{code}", header, offset)[0])


def _real(header: bytes, offset: int) -> float:
    return float(struct.unpack_from("<f", header, offset)[0])


def _text(header: bytes, offset: int, length: int) -> str:
    return header[offset:offset + length].decode(
        "ascii", errors="replace").rstrip("\x00 ")


def _bin(value: int, length: int) -> int | None:
    if value < 0:
        return None
    # Some files use the histogram length as an exclusive last-good value.
    return min(value, length - 1)


def _instrument(path: Path) -> str:
    match = _INSTRUMENT.search(path.stem)
    return match.group(1).upper() if match else ""


def _suggested_configuration(
    instrument: str,
    detectors: list[DetectorInfo],
) -> dict:
    by_name: dict[str, list[int]] = {}
    for detector in detectors:
        by_name.setdefault(detector.name.strip().casefold(), []).append(
            detector.number)
    labelled_forward = by_name.get("forw", []) + by_name.get("forward", [])
    labelled_backward = by_name.get("back", []) + by_name.get("backward", [])
    if len(labelled_forward) == len(labelled_backward) == 1:
        reverse_labels = "gpd" not in instrument.casefold()
        return {
            "fittype": 2,
            "forward": (
                labelled_backward if reverse_labels else labelled_forward),
            "backward": (
                labelled_forward if reverse_labels else labelled_backward),
            "packing": 20,
        }
    return {}


class PsiBinLoader:
    """Load read-only PSI-BIN files into the common raw-histogram model."""

    name = "PSI-BIN"

    @staticmethod
    def can_load(path: Path) -> bool:
        if path.suffix.lower() != ".bin":
            return False
        with path.open("rb") as stream:
            return stream.read(2) == _FORMAT_ID

    @staticmethod
    def load(path: Path) -> HistogramRun:
        raw = path.read_bytes()
        if len(raw) < _HEADER_SIZE or raw[:2] != _FORMAT_ID:
            raise UnsupportedFormatError(f"{path.name} is not a PSI-BIN file")
        if len(raw) % _RECORD_SIZE:
            raise UnsupportedFormatError(
                "PSI-BIN size is not an integer number of 512-byte records")

        header = raw[:_HEADER_SIZE]
        run_number = _integer(header, 6)
        histogram_length = _integer(header, 28)
        histogram_count = _integer(header, 30)
        record_count = _integer(header, 128)
        record_bins = _integer(header, 130)
        records_per_histogram = _integer(header, 132)
        histograms_per_record = _integer(header, 134)

        if not 1 <= histogram_count <= _MAX_HISTOGRAMS:
            raise UnsupportedFormatError(
                f"PSI-BIN histogram count {histogram_count} is unsupported")
        if histogram_length < 2 or record_bins <= 0 or records_per_histogram <= 0:
            raise UnsupportedFormatError("PSI-BIN has invalid histogram geometry")
        if histograms_per_record != 1:
            raise UnsupportedFormatError(
                "PSI-BIN files with multiple histograms per record are unsupported")
        stored_bins = record_bins * records_per_histogram
        if histogram_length > stored_bins:
            raise UnsupportedFormatError(
                "PSI-BIN histogram length exceeds its allocated records")
        required_records = histogram_count * records_per_histogram
        if record_count < required_records:
            raise UnsupportedFormatError(
                "PSI-BIN does not contain enough histogram records")
        payload_size = record_count * record_bins * 4
        if len(raw) < _HEADER_SIZE + payload_size:
            raise UnsupportedFormatError("PSI-BIN histogram payload is truncated")

        flat = np.frombuffer(
            raw,
            dtype="<i4",
            count=record_count * record_bins,
            offset=_HEADER_SIZE,
        )
        counts = np.vstack([
            flat[position * stored_bins:
                 position * stored_bins + histogram_length]
            for position in range(histogram_count)
        ]).astype(float)
        if np.any(counts < 0):
            raise UnsupportedFormatError("PSI-BIN contains negative histogram counts")

        bin_width_us = _real(header, 1012)
        if not np.isfinite(bin_width_us) or bin_width_us <= 0:
            tdc_resolution = _integer(header, 2)
            bin_width_us = 0.125 * 625e-6 * 2.0 ** tdc_resolution
        if not np.isfinite(bin_width_us) or bin_width_us <= 0:
            raise UnsupportedFormatError("PSI-BIN has no valid time-bin width")

        t0 = [_integer(header, 458 + position * 2)
              for position in range(histogram_count)]
        first_good = [_integer(header, 490 + position * 2)
                      for position in range(histogram_count)]
        last_good = [_integer(header, 522 + position * 2)
                     for position in range(histogram_count)]
        labels = [_text(header, 948 + position * 4, 4)
                  for position in range(histogram_count)]
        events = [_integer(header, 296 + position * 4, size=4)
                  for position in range(histogram_count)]
        real_t0 = [_real(header, 792 + position * 4)
                   for position in range(histogram_count)]
        detectors = [
            DetectorInfo(
                number=position + 1,
                name=labels[position] or f"Detector {position + 1}",
                t0_bin=_bin(t0[position], histogram_length),
                first_good_bin=_bin(first_good[position], histogram_length),
                last_good_bin=_bin(last_good[position], histogram_length),
                metadata={
                    "events": events[position],
                    "real_t0_us": real_t0[position],
                },
            )
            for position in range(histogram_count)
        ]

        temperatures = [_real(header, 716 + position * 4)
                        for position in range(4)]
        temperature_errors = [_real(header, 738 + position * 4)
                              for position in range(4)]
        valid_temperatures = [
            (value, error)
            for value, error in zip(temperatures, temperature_errors)
            if np.isfinite(value) and value > 0
        ]
        header_temperature = _text(header, 148, 10)
        if valid_temperatures:
            value, error = valid_temperatures[0]
            temperature = f"{value:g} +- {max(error, 0.0):g} K"
        else:
            temperature = header_temperature

        start_date = _text(header, 218, 9)
        stop_date = _text(header, 227, 9)
        start_clock = _text(header, 236, 8)
        stop_clock = _text(header, 244, 8)
        comment = _text(header, 860, 62)
        instrument = _instrument(path)
        metadata = {
            "source": str(path),
            "reader": PsiBinLoader.name,
            "facility": "PSI",
            "instrument": instrument,
            "format": "PSI-BIN",
            "run_number": run_number,
            "title": comment,
            "sample": _text(header, 138, 10),
            "temperature": temperature,
            "field": _text(header, 158, 10),
            "start_time": " ".join(
                value for value in (start_date, start_clock) if value),
            "end_time": " ".join(
                value for value in (stop_date, stop_clock) if value),
            "time_resolution_us": bin_width_us,
            "suggested_run_configuration": _suggested_configuration(
                instrument, detectors),
            "file_header": {
                "format_id": _FORMAT_ID.decode("ascii"),
                "setup": _text(header, 178, 10),
                "orientation": _text(header, 168, 10),
                "header_temperature": header_temperature,
                "comment": comment,
                "total_events": _integer(header, 424, size=4),
                "temperatures_K": temperatures,
                "temperature_errors_K": temperature_errors,
            },
        }
        time = np.arange(histogram_length, dtype=float) * bin_width_us
        return HistogramRun(time, counts, metadata, detectors)
