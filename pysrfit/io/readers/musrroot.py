"""Reader for PSI's documented MusrRoot ROOT-file structure."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re

import numpy as np

from ..base import DetectorInfo, HistogramRun, UnsupportedFormatError


_HEADER_LINE = re.compile(r"^\s*\d+\s*-\s*([^:]+):\s*(.*?)\s+-@\d+\s*$")
_DECAY_NAME = re.compile(r"^hDecay(\d{3})$")
_DETECTOR_NAME = re.compile(r"^Detector(\d+)$", flags=re.IGNORECASE)


def _name(obj) -> str:
    try:
        return str(obj.member("fName"))
    except (AttributeError, KeyError):
        path = getattr(obj, "path", ())
        if path:
            return str(path[-1])
        raise UnsupportedFormatError("MusrRoot contains an unnamed object")


def _entries(container) -> list:
    """Return immediate children from old TFolder or new TDirectory files."""
    if hasattr(container, "items"):
        # Using items rather than indexing by name is important because header
        # strings may themselves contain semicolons, which ROOT also uses to
        # separate object names from cycle numbers.
        return [value for _key, value in container.items(
            recursive=False, cycle=False)]
    return list(container)


def _children(folder) -> list:
    try:
        return list(folder.member("fFolders"))
    except (AttributeError, KeyError):
        return _entries(folder)


def _named_child(folder, name: str):
    for child in _children(folder):
        if _name(child) == name:
            return child
    raise UnsupportedFormatError(f"MusrRoot is missing {name!r}")


def _fields(strings) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in _entries(strings):
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


def _detector_number(record, fields: dict[str, str], position: int) -> int:
    """Use the detector object's identity, which is authoritative for LEM.

    Some LEM headers repeat the local ``Histo Number`` for a second external-
    field state, while their Detector041...Detector068 object names and
    hDecay041...hDecay068 histograms remain unique.
    """
    match = _DETECTOR_NAME.fullmatch(_name(record))
    if match:
        return int(match.group(1))
    return _integer(fields, "Histo Number", position + 1)


def _normalized_name(value: str) -> str:
    return " ".join(value.casefold().split())


def _layout_matches(
    detectors: list[DetectorInfo],
    expected: dict[int, str],
) -> bool:
    actual = {
        detector.number: _normalized_name(detector.name)
        for detector in detectors
    }
    normalized_expected = {
        number: _normalized_name(name) for number, name in expected.items()
    }
    return actual == normalized_expected


def _hal_layout() -> dict[int, str]:
    return {1: "MV", **{number + 1: f"F{number}" for number in range(1, 9)}}


def _lem_layout() -> dict[int, str]:
    directions = (
        "Left D(F)", "Top D(F)", "Right D(F)", "Bottom D(F)",
        "Left U(B)", "Top U(B)", "Right U(B)", "Bottom U(B)",
    )
    result = {}
    for offset, external_state, post_pileup in (
        (0, "OFF", False),
        (20, "OFF", True),
        (40, "ON", False),
        (60, "ON", True),
    ):
        for position, direction in enumerate(directions, 1):
            correction = " PostPileUp rej." if post_pileup else ""
            result[offset + position] = (
                f"e+ {direction}{correction}, EXT. {external_state}")
    return result


def _suggested_run_configuration(
    instrument: str,
    detectors: list[DetectorInfo],
) -> dict:
    """Return a conservative initial configuration for known PSI instruments."""
    normalized = instrument.casefold()
    by_name: dict[str, list[int]] = {}
    for detector in detectors:
        by_name.setdefault(_normalized_name(detector.name), []).append(
            detector.number)

    if "hal9500" in normalized or "hifi" in normalized:
        # HAL's azimuthal F1...F8 counters are normally fitted as separate
        # histograms. Start with F1 and preserve the fine time resolution.
        if _layout_matches(detectors, _hal_layout()):
            return {
                "fittype": 0, "forward": [2], "packing": 5,
                "data_stop_us": 8.0,
            }

    if normalized == "lem" or normalized.endswith("_lem"):
        # Standard LEM left/right grouping: downstream + upstream halves.
        if _layout_matches(detectors, _lem_layout()):
            return {
                "fittype": 2,
                "forward": [1, 5],
                "backward": [3, 7],
                "packing": 100,
                "data_stop_us": 6.0,
            }

    labelled_forward = by_name.get("forw", []) + by_name.get("forward", [])
    labelled_backward = by_name.get("back", []) + by_name.get("backward", [])
    if len(labelled_forward) == 1 and len(labelled_backward) == 1:
        # GPS and FLAME analysis assign the counters opposite to their
        # hardware-facing Forw/Back labels. GPD uses the labelled roles.
        reverse_labels = "gpd" not in normalized
        suggestion = {
            "fittype": 2,
            "forward": (
                labelled_backward if reverse_labels else labelled_forward),
            "backward": (
                labelled_forward if reverse_labels else labelled_backward),
            "packing": 20,
        }
        if "gpd" in normalized or normalized == "flame":
            suggestion["data_stop_us"] = 8.0
        return suggestion
    return {}


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
                "MusrRoot support requires a working uproot installation; "
                "reinstall PySRFit with 'python -m pip install --force-reinstall pysrfit'") from exc

        try:
            with uproot.open(path) as root:
                header = root["RunHeader"]
                histograms = root["histos"]
                run_info = _fields(_named_child(header, "RunInfo"))
                detector_records = _entries(
                    _named_child(header, "DetectorInfo"))
                detector_fields = [_fields(record) for record in detector_records]
                decay_folder = _named_child(histograms, "DecayAnaModule")
                detectors = [
                    DetectorInfo(
                        number=_detector_number(
                            detector_records[position], fields, position),
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

                # New FLAME directory-format files currently use zero as an
                # unset t0/first-good sentinel. Estimate the prompt channel so
                # the initial configuration is usable and remains editable in
                # the visual correction dialog.
                repaired_detectors = []
                for detector, values in zip(detectors, counts):
                    if detector.t0_bin is not None and detector.t0_bin <= 0:
                        prompt = int(np.argmax(values))
                        metadata = dict(detector.metadata)
                        metadata["Header Time Zero Bin"] = detector.t0_bin
                        metadata["Time Zero Source"] = "histogram maximum"
                        repaired_detectors.append(replace(
                            detector,
                            t0_bin=float(prompt),
                            first_good_bin=max(
                                prompt + 1, detector.first_good_bin or 0),
                            metadata=metadata,
                        ))
                    else:
                        repaired_detectors.append(detector)
                detectors = repaired_detectors

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
            "suggested_run_configuration": _suggested_run_configuration(
                run_info.get("Instrument", ""), detectors),
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

