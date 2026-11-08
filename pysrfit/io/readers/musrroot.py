"""Reader for PSI's documented MusrRoot ROOT-file structure."""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np

from ..base import DetectorInfo, HistogramRun, UnsupportedFormatError


_HEADER_LINE = re.compile(r"^\s*\d+\s*-\s*([^:]+):\s*(.*?)\s+-@\d+\s*$")
_DECAY_NAME = re.compile(r"^hDecay(\d{3})$")


def _name(obj) -> str:
    return str(obj.member("fName"))


def _children(folder) -> list:
    return list(folder.member("fFolders"))


def _named_child(folder, name: str):
    for child in _children(folder):
        if _name(child) == name:
            return child
    raise UnsupportedFormatError(f"MusrRoot is missing {name!r}")


def _fields(strings) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in strings:
        match = _HEADER_LINE.match(str(item))
        if match:
            result[match.group(1).strip()] = match.group(2).strip()
    return result


def _integer(fields: dict[str, str], key: str, default=None):
    value = fields.get(key)
    return default if value is None else int(float(value))


def _number(fields: dict[str, str], key: str, default=None):
    value = fields.get(key)
    return default if value is None else float(value.split()[0])


class MusrRootLoader:
    name = "MusrRoot"

    @staticmethod
    def can_load(path: Path) -> bool:
        if path.suffix.lower() != ".root":
            return False
        with path.open("rb") as stream:
            return stream.read(4) == b"root"

    @staticmethod
    def load(path: Path) -> HistogramRun:
        try:
            import uproot
        except ImportError as exc:
            raise UnsupportedFormatError(
                "MusrRoot support requires uproot; install with 'pip install pysrfit[root]'") from exc

        try:
            with uproot.open(path) as root:
                header = root["RunHeader"]
                histograms = root["histos"]
                run_info = _fields(_named_child(header, "RunInfo"))
                detector_fields = [
                    _fields(array) for array in _named_child(header, "DetectorInfo")
                ]
                decay_folder = _named_child(histograms, "DecayAnaModule")
                detectors = [
                    DetectorInfo(
                        number=_integer(fields, "Histo Number", position + 1),
                        name=fields.get("Name", f"Detector {position + 1}"),
                        t0_bin=_number(fields, "Time Zero Bin"),
                        first_good_bin=_integer(fields, "First Good Bin"),
                        last_good_bin=_integer(fields, "Last Good Bin"),
                        metadata={
                            key: value for key, value in fields.items()
                            if key not in {"Name", "Histo Number", "Time Zero Bin", "First Good Bin", "Last Good Bin"}
                        },
                    )
                    for position, fields in enumerate(detector_fields)
                ]

                decay_objects = []
                for obj in _children(decay_folder):
                    match = _DECAY_NAME.match(_name(obj))
                    if match:
                        decay_objects.append((int(match.group(1)), obj))
                decay_objects.sort(key=lambda item: item[0])
                if not decay_objects:
                    raise UnsupportedFormatError("MusrRoot contains no hDecayNNN histograms")
                detector_by_number = {detector.number: detector for detector in detectors}
                histogram_numbers = [number for number, _ in decay_objects]
                if set(histogram_numbers) != set(detector_by_number):
                    raise UnsupportedFormatError(
                        "MusrRoot histogram numbers do not match its DetectorInfo records")
                detectors = [detector_by_number[number] for number in histogram_numbers]
                counts = [np.asarray(obj.values(flow=False), dtype=float) for _, obj in decay_objects]
                lengths = {len(values) for values in counts}
                if len(lengths) != 1:
                    raise UnsupportedFormatError(
                        "MusrRoot decay histograms have different lengths; ragged runs are not supported")
                if len(detectors) != len(counts):
                    raise UnsupportedFormatError(
                        f"MusrRoot has {len(counts)} histograms but {len(detectors)} detector records")

                resolution = run_info.get("Time Resolution", "")
                match = re.match(r"\s*([0-9.eE+-]+)\s*(ns|us|µs|ms|s)\b", resolution)
                if not match:
                    raise UnsupportedFormatError(
                        f"MusrRoot time resolution is missing or unsupported: {resolution!r}")
                scale = {"ns": 1e-3, "us": 1.0, "µs": 1.0, "ms": 1e3, "s": 1e6}[match.group(2)]
                channel_width_us = float(match.group(1)) * scale
                time = np.arange(next(iter(lengths)), dtype=float) * channel_width_us
        except UnsupportedFormatError:
            raise
        except Exception as exc:
            raise UnsupportedFormatError(f"Cannot read MusrRoot file {path.name}: {exc}") from exc

        metadata = {
            "source": str(path), "reader": MusrRootLoader.name,
            "facility": run_info.get("Laboratory", ""),
            "instrument": run_info.get("Instrument", ""),
            "title": run_info.get("Run Title", ""),
            "run_number": _integer(run_info, "Run Number"),
            "sample": run_info.get("Sample Name", ""),
            "temperature": run_info.get("Sample Temperature", ""),
            "field": run_info.get("Sample Magnetic Field", ""),
            "time_resolution_us": channel_width_us,
            "format": "MUSR-ROOT",
            "file_header": {
                key: value for key, value in run_info.items()
                if key not in {
                    "Laboratory", "Instrument", "Run Title", "Run Number",
                    "Sample Name", "Sample Temperature", "Sample Magnetic Field",
                    "Time Resolution",
                }
            },
        }
        return HistogramRun(time, np.vstack(counts), metadata, detectors)

