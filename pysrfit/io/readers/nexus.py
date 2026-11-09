"""Reader for modern HDF5 NeXus muon data and generic NXdata spectra."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..base import DetectorInfo, HistogramRun, Spectrum, UnsupportedFormatError


_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
_HDF4_MAGIC = b"\x0e\x03\x13\x01"
_SUFFIXES = {".nxs", ".nexus", ".h5", ".hdf5"}


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="replace").rstrip("\x00")
    if isinstance(value, (list, tuple, np.ndarray)):
        array = np.asarray(value)
        return _text(array.flat[0]) if array.size else ""
    return str(value).rstrip("\x00")


def _scalar(value: Any, default=None):
    if value is None:
        return default
    array = np.asarray(value)
    return default if not array.size else array.flat[0].item()


def _nx_class(obj) -> str:
    return _text(obj.attrs.get("NX_class", obj.attrs.get("NXclass", "")))


def _scale_to_us(unit: str) -> float | None:
    normalized = unit.casefold().replace("µ", "u")
    normalized = "".join(character for character in normalized if character.isalnum())
    return {
        "us": 1.0,
        "microsecond": 1.0,
        "microseconds": 1.0,
        "ns": 1e-3,
        "nanosecond": 1e-3,
        "nanoseconds": 1e-3,
        "ms": 1e3,
        "millisecond": 1e3,
        "milliseconds": 1e3,
        "s": 1e6,
        "second": 1e6,
        "seconds": 1e6,
    }.get(normalized)


def _entry_for(group):
    current = group
    while current.name != "/":
        if _nx_class(current) == "NXentry":
            return current
        current = current.parent
    return group.parent


def _first_group_by_class(group, wanted: str):
    import h5py

    matches = []

    def collect(_, obj):
        if isinstance(obj, h5py.Group) and _nx_class(obj) == wanted:
            matches.append(obj)

    group.visititems(collect)
    return matches[0] if matches else None


def _child(entry, path: str):
    try:
        return entry[path]
    except KeyError:
        return None


def _value(entry, path: str, default=None):
    dataset = _child(entry, path)
    return default if dataset is None else _scalar(dataset[()], default)


def _string(entry, path: str, default="") -> str:
    value = _value(entry, path, None)
    return default if value is None else _text(value).strip()


def _quantity(entry, path: str, *, default_unit: str = "") -> str:
    dataset = _child(entry, path)
    if dataset is None:
        return ""
    value = _scalar(dataset[()], None)
    if value is None:
        return ""
    unit = _text(dataset.attrs.get("units", default_unit)).strip()
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = f"{float(value):g}"
    else:
        number = _text(value).strip()
    return f"{number} {unit}".strip()


def _signal_dataset(group):
    import h5py

    signal_name = _text(group.attrs.get("signal", "")).strip()
    if signal_name and signal_name in group and isinstance(group[signal_name], h5py.Dataset):
        return group[signal_name]
    for obj in group.values():
        if isinstance(obj, h5py.Dataset) and _scalar(obj.attrs.get("signal"), 0) == 1:
            return obj
    if "counts" in group and isinstance(group["counts"], h5py.Dataset):
        return group["counts"]
    return None


def _time_dataset(group):
    import h5py

    for name in ("raw_time", "time", "time_us", "time_centers", "time_bins"):
        if name in group and isinstance(group[name], h5py.Dataset):
            return group[name]
    return None


def _per_detector(source, count: int, fallback=None) -> list[Any] | None:
    if source is None:
        source = fallback
    if source is None:
        return None
    values = np.asarray(source).reshape(-1)
    if len(values) == 1:
        return [values[0].item()] * count
    if len(values) == count:
        return [value.item() for value in values]
    return None


def _bin_value(value, maximum: int) -> int | None:
    if value is None or not np.isfinite(float(value)):
        return None
    return min(max(int(round(float(value))), 0), maximum)


def _grouping_configuration(detector_group, numbers: list[int]):
    grouping_set = _child(detector_group, "grouping") if detector_group is not None else None
    if grouping_set is None or not bool(_scalar(grouping_set.attrs.get("available"), 1)):
        return None, {}, {}
    grouping = np.asarray(grouping_set).reshape(-1)
    if len(grouping) != len(numbers):
        return None, {}, {}
    names = [name.strip() for name in _text(
        grouping_set.attrs.get("name_groups", "")).split(";")]
    unique = sorted({int(value) for value in grouping if int(value) > 0})
    if not unique or max(unique) > len(names):
        return None, {}, {}

    labels: dict[int, str] = {}
    members: dict[str, list[int]] = {}
    counters: dict[str, int] = {}
    for number, group_number in zip(numbers, grouping):
        group_number = int(group_number)
        if group_number <= 0:
            continue
        label = names[group_number - 1]
        normalized = label.casefold()
        counters[normalized] = counters.get(normalized, 0) + 1
        labels[number] = f"{label} {counters[normalized]}"
        members.setdefault(normalized, []).append(number)

    forward = members.get("fwd", members.get("forward", []))
    backward = members.get("bwd", members.get("backward", []))
    if (len(members) == 2 and forward and backward
            and len(forward) == len(backward)):
        suggestion = {
            "fittype": 2,
            "forward": forward,
            "backward": backward,
            "packing": 1,
            "data_stop_us": 8.0,
        }
    else:
        suggestion = {}
    return grouping, labels, suggestion


def _known_instrument_configuration(
    facility: str,
    instrument: str,
    numbers: list[int],
) -> dict:
    # Complete detector layout from the bundled EMU NeXus-2 reference and
    # musrfit's matching example. A changed count/numbering is left untouched.
    if (facility.casefold() == "isis" and instrument.casefold() == "emu"
            and numbers == list(range(1, 65))):
        return {
            "fittype": 2,
            "forward": list(range(1, 17)),
            "backward": list(range(17, 33)),
            "backgr.fix": [0.0, 0.0],
            "packing": 1,
            "data_stop_us": 14.0,
        }
    return {}


def _configuration_has_usable_start(
    signal: np.ndarray,
    detectors: list[DetectorInfo],
    suggestion: dict,
) -> bool:
    """Reject file-supplied grouping when its good-bin metadata points at zeros."""
    if not suggestion:
        return False
    by_number = {detector.number: position for position, detector in enumerate(detectors)}
    for role in ("forward", "backward"):
        numbers = suggestion.get(role, [])
        if not numbers or any(number not in by_number for number in numbers):
            return False
        selected = [detectors[by_number[number]] for number in numbers]
        start = selected[0].first_good_bin
        if start is None:
            return False
        stop = min(start + 15, signal.shape[1] - 1)
        indices = [by_number[number] for number in numbers]
        if float(np.sum(signal[indices, start:stop + 1])) <= 0:
            return False
    return True


def _discard_unusable_timing(
    detectors: list[DetectorInfo],
) -> list[DetectorInfo]:
    """Keep a trusted layout while preventing fallback to bad timing metadata."""
    result = []
    for detector in detectors:
        metadata = dict(detector.metadata)
        if detector.t0_bin is not None:
            metadata["reported_t0_bin"] = detector.t0_bin
        if detector.first_good_bin is not None:
            metadata["reported_first_good_bin"] = detector.first_good_bin
        result.append(DetectorInfo(
            number=detector.number,
            name=detector.name,
            t0_bin=None,
            first_good_bin=None,
            last_good_bin=detector.last_good_bin,
            metadata=metadata,
        ))
    return result


def _detector_metadata(group, signal, count: int):
    entry = _entry_for(group)
    detector_group = _child(entry, "instrument/detector_1")
    if detector_group is None:
        detector_group = _first_group_by_class(entry, "NXdetector")

    spectrum_set = _child(group, "spectrum_index")
    numbers_raw = _per_detector(
        spectrum_set[()] if spectrum_set is not None else None, count)
    if numbers_raw is None:
        numbers = list(range(1, count + 1))
    else:
        numbers = [int(value) for value in numbers_raw]
        if min(numbers, default=0) < 1 or len(set(numbers)) != count:
            numbers = list(range(1, count + 1))

    def detector_values(name: str):
        dataset = _child(detector_group, name) if detector_group is not None else None
        return _per_detector(
            dataset[()] if dataset is not None else None,
            count,
            signal.attrs.get(name),
        )

    t0 = detector_values("time_zero_bin") or detector_values("t0_bin")
    first = detector_values("first_good_bin")
    last = detector_values("last_good_bin")
    _grouping, labels, grouping_suggestion = _grouping_configuration(
        detector_group, numbers)
    maximum = signal.shape[-1] - 1
    detectors = [
        DetectorInfo(
            number=number,
            name=labels.get(number, f"Spectrum {number}"),
            t0_bin=_bin_value(None if t0 is None else t0[position], maximum),
            first_good_bin=_bin_value(
                None if first is None else first[position], maximum),
            last_good_bin=_bin_value(
                None if last is None else last[position], maximum),
        )
        for position, number in enumerate(numbers)
    ]
    return entry, detectors, numbers, grouping_suggestion


def _load_nxdata(group, path: Path) -> Spectrum | HistogramRun | None:
    signal_set = _signal_dataset(group)
    time_set = _time_dataset(group)
    if signal_set is None or time_set is None:
        return None
    scale = _scale_to_us(_text(time_set.attrs.get("units", "")).strip())
    if scale is None:
        raise UnsupportedFormatError(
            f"NeXus '{time_set.name}' time unit "
            f"{_text(time_set.attrs.get('units', ''))!r} is absent or unknown")

    signal = np.asarray(signal_set)
    period_count = 1
    if signal.ndim == 3:
        period_count = signal.shape[0]
        if period_count != 1:
            raise UnsupportedFormatError(
                f"NeXus contains {period_count} periods; select/add-runs semantics "
                "must be configured before multi-period data can be loaded")
        signal = signal[0]

    raw_time = np.asarray(time_set, dtype=float).reshape(-1) * scale
    bins = signal.shape[-1]
    if len(raw_time) >= 2:
        # Facility files commonly store an exactly uniform axis as float32,
        # whose quantization makes late-bin differences appear irregular.
        # Reconstruct the intended uniform grid only when every interval is
        # already close to the endpoint-derived spacing.
        width = (raw_time[-1] - raw_time[0]) / (len(raw_time) - 1)
        differences = np.diff(raw_time)
        if width > 0 and np.allclose(differences, width, rtol=1e-3, atol=1e-12):
            raw_time = raw_time[0] + np.arange(len(raw_time), dtype=float) * width
    if len(raw_time) == bins + 1:
        time = (raw_time[:-1] + raw_time[1:]) / 2.0
    elif len(raw_time) == bins:
        time = raw_time
    else:
        return None
    if len(time) < 2 or np.any(np.diff(time) <= 0):
        raise UnsupportedFormatError("NeXus raw time axis is not strictly increasing")

    if signal.ndim == 1 and "asym" in signal_set.name.casefold():
        error_name = _text(group.attrs.get("errors", ""))
        if error_name in group:
            return Spectrum(time, signal, np.asarray(group[error_name]), {
                "source": str(path), "reader": NexusLoader.name,
                "nxdata": group.name, "signal": signal_set.name,
                "format": "NEXUS",
            })
        raise UnsupportedFormatError("NeXus asymmetry lacks an error array")
    if signal.ndim != 2:
        return None

    entry, detectors, numbers, grouping_suggestion = _detector_metadata(
        group, signal_set, signal.shape[0])
    instrument = _string(entry, "instrument/name")
    facility = _string(entry, "instrument/source/name")
    if not facility and instrument.casefold() in {"emu", "hifi", "musr", "argus"}:
        facility = "ISIS"
    suggestion = grouping_suggestion or _known_instrument_configuration(
        facility, instrument, numbers)
    if grouping_suggestion and not _configuration_has_usable_start(
            signal, detectors, grouping_suggestion):
        # The grouping itself is still useful. Keep it, but do not let the
        # session silently fall back to t0/first-good metadata whose proposed
        # data region contains no counts.
        suggestion = dict(grouping_suggestion)
        suggestion.pop("data_stop_us", None)
        suggestion["manual_timing_required"] = True
        detectors = _discard_unusable_timing(detectors)
    default_temperature_unit = "K" if facility.casefold() == "j-parc" else ""
    default_field_unit = "G" if facility.casefold() == "j-parc" else ""
    temperature_path = (
        "sample/temperature" if _child(entry, "sample/temperature") is not None
        else "sample/temperature_1")
    field_path = (
        "sample/magnetic_field" if _child(entry, "sample/magnetic_field") is not None
        else "sample/magnetic_field_1")
    metadata = {
        "source": str(path),
        "reader": NexusLoader.name,
        "facility": facility,
        "instrument": instrument,
        "format": "NEXUS",
        "definition": _string(entry, "definition"),
        "run_number": _value(entry, "run_number"),
        "title": _string(entry, "title"),
        "sample": _string(entry, "sample/name"),
        "temperature": _quantity(
            entry, temperature_path, default_unit=default_temperature_unit),
        "field": _quantity(entry, field_path, default_unit=default_field_unit),
        "start_time": _string(entry, "start_time"),
        "end_time": _string(entry, "end_time"),
        "experiment_identifier": _string(entry, "experiment_identifier"),
        "nxdata": group.name,
        "signal": signal_set.name,
        "periods": period_count,
        "time_resolution_us": float(np.median(np.diff(time))),
        "suggested_run_configuration": suggestion,
    }
    return HistogramRun(time, signal.astype(float, copy=False), metadata, detectors)


class NexusLoader:
    name = "NeXus NXdata"

    @staticmethod
    def can_load(path: Path) -> bool:
        return path.suffix.lower() in _SUFFIXES

    @staticmethod
    def load(path: Path) -> Spectrum | HistogramRun:
        with path.open("rb") as stream:
            signature = stream.read(8)
        if signature.startswith(_HDF4_MAGIC):
            raise UnsupportedFormatError(
                "Legacy HDF4 NeXus is not supported. Use a modern HDF5 NeXus "
                "export or convert the run before importing it.")
        if signature != _HDF5_MAGIC:
            raise UnsupportedFormatError(f"{path.name} is not an HDF5 NeXus file")
        try:
            import h5py
        except (ImportError, ValueError) as exc:
            raise UnsupportedFormatError(
                "NeXus support requires a compatible h5py installation; "
                "reinstall PySRFit with 'python -m pip install --force-reinstall pysrfit'") from exc

        try:
            with h5py.File(path, "r") as root:
                groups = []

                def collect(_, obj):
                    if isinstance(obj, h5py.Group) and _nx_class(obj) == "NXdata":
                        groups.append(obj.name)

                root.visititems(collect)
                errors = []
                for name in groups:
                    try:
                        loaded = _load_nxdata(root[name], path)
                    except UnsupportedFormatError as exc:
                        errors.append(str(exc))
                        continue
                    if loaded is not None:
                        return loaded
                if errors:
                    raise UnsupportedFormatError(errors[0])
        except UnsupportedFormatError:
            raise
        except Exception as exc:
            raise UnsupportedFormatError(
                f"Cannot read NeXus file {path.name}: {exc}") from exc
        raise UnsupportedFormatError(
            "No recognized NXdata counts(detector,time) or asymmetry(time) "
            "with explicit time units")
