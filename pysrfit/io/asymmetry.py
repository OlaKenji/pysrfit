"""Prepare aligned asymmetry or raw-count data from facility-neutral histograms.

The RUN vocabulary follows musrfit: detector numbers are one-based, detector
groups may use ranges, t0 values align grouped histograms, and fittype 0/2
select raw-histogram/asymmetry preparation respectively.
"""
from __future__ import annotations

from collections.abc import Sequence
import re

import numpy as np

from .base import CountSpectrum, HistogramRun, Spectrum
from ..models import RunConfig


_INTEGER_RANGE = re.compile(r"^(\d+)-(\d+)$")


def detector_group(data: HistogramRun, references: Sequence[str | int]) -> list[int]:
    """Resolve musrfit-style detector numbers/ranges or detector names to array indices."""
    expanded: list[str | int] = []
    for reference in references:
        match = _INTEGER_RANGE.match(str(reference).strip())
        if match:
            start, stop = map(int, match.groups())
            if stop < start:
                raise ValueError(f"Descending detector range {reference!r}")
            expanded.extend(range(start, stop + 1))
        else:
            expanded.append(reference)
    if not expanded:
        raise ValueError("At least one detector must be selected")
    indices = [data.detector_index(reference) for reference in expanded]
    if len(set(indices)) != len(indices):
        raise ValueError("A detector may appear only once in a group")
    return indices


def _numbers(run: RunConfig, key: str, length: int | None = None, *, default=None) -> list[float]:
    tokens = run.tokens(key)
    if not tokens:
        if default is not None:
            return list(default)
        raise ValueError(f"RUN requires '{key}'")
    values = [float(token) for token in tokens]
    if length is not None and len(values) < length:
        raise ValueError(f"RUN '{key}' requires at least {length} entries")
    return values


def _metadata_t0(data: HistogramRun, indices: Sequence[int]) -> list[float]:
    result = [data.detectors[index].t0_bin for index in indices]
    if any(value is None for value in result):
        missing = [data.detectors[index].name for index, value in zip(indices, result) if value is None]
        raise ValueError(f"No t0 supplied and detector metadata has no t0 for {', '.join(missing)}")
    return [float(value) for value in result]


def _t0_groups(
    data: HistogramRun,
    run: RunConfig,
    forward: Sequence[int],
    backward: Sequence[int] = (),
) -> tuple[list[float], list[float]]:
    tokens = _numbers(run, "t0", default=[])
    if not tokens:
        return _metadata_t0(data, forward), _metadata_t0(data, backward)
    if not backward:
        if len(tokens) != len(forward):
            raise ValueError("Single-histogram t0 needs one value per grouped detector")
        return tokens, []
    if len(forward) != len(backward):
        raise ValueError("Asymmetry detector groups currently require equal sizes")
    if len(tokens) != 2 * len(forward):
        raise ValueError("Asymmetry t0 order is forward1 backward1 [forward2 backward2 ...]")
    return tokens[0::2], tokens[1::2]


def _require_integer_t0(values: Sequence[float]) -> list[int]:
    if any(abs(value - round(value)) > 1e-8 for value in values):
        raise NotImplementedError("Fractional-bin t0 needs validated interpolation and error propagation")
    result = [int(round(value)) for value in values]
    if any(value < 0 for value in result):
        raise ValueError("t0 cannot be negative")
    return result


def _aligned_group(
    data: HistogramRun,
    indices: Sequence[int],
    t0_values: Sequence[float],
    reference_channels: np.ndarray,
) -> np.ndarray:
    t0 = _require_integer_t0(t0_values)
    reference_t0 = t0[0]
    result = np.zeros(len(reference_channels), dtype=float)
    for index, detector_t0 in zip(indices, t0):
        channels = reference_channels + detector_t0 - reference_t0
        if len(channels) and (channels[0] < 0 or channels[-1] >= data.counts.shape[1]):
            raise ValueError("Detector grouping/t0 alignment runs outside a histogram")
        result += data.counts[index, channels]
    return result


def _common_offset_bounds(
    data: HistogramRun,
    groups: Sequence[tuple[Sequence[int], Sequence[float]]],
) -> tuple[int, int]:
    lower, upper = 0, data.counts.shape[1] - 1
    for _, values in groups:
        t0 = _require_integer_t0(values)
        lower = max(lower, max(-value for value in t0))
        upper = min(upper, min(data.counts.shape[1] - 1 - value for value in t0))
    return lower, upper


def _range_from_metadata(data: HistogramRun, index: int) -> tuple[int, int] | None:
    detector = data.detectors[index]
    if detector.first_good_bin is None or detector.last_good_bin is None:
        return None
    return detector.first_good_bin, min(detector.last_good_bin, data.counts.shape[1] - 1)


def _pack(values: np.ndarray, packing: int) -> tuple[np.ndarray, int]:
    if packing < 1:
        raise ValueError("packing must be at least one")
    count = (len(values) // packing) * packing
    if not count:
        raise ValueError("Packing is larger than the available data")
    return values[:count].reshape(-1, packing).sum(axis=1), count


def _background(
    data: HistogramRun,
    indices: Sequence[int],
    t0: Sequence[float],
    start: int,
    stop: int,
) -> tuple[float, float]:
    if not 0 <= start <= stop < data.counts.shape[1]:
        raise ValueError("Background bin range is outside the reference histogram")
    channels = np.arange(start, stop + 1, dtype=int)
    values = _aligned_group(data, indices, t0, channels)
    mean = float(np.mean(values))
    variance_of_mean = float(np.sum(values)) / len(values) ** 2
    return mean, variance_of_mean


def asymmetry_from_components(
    spectrum: Spectrum,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Recalculate asymmetry and uncertainty from cached detector components."""
    components = spectrum.metadata.get("asymmetry_components")
    if not isinstance(components, dict):
        raise ValueError("Asymmetry spectrum has no detector components for floating alpha")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Alpha must be positive")
    forward = np.asarray(components["forward"], dtype=float)
    backward = np.asarray(components["backward"], dtype=float)
    forward_variance = np.asarray(components["forward_variance"], dtype=float)
    backward_variance = np.asarray(components["backward_variance"], dtype=float)
    denominator = alpha * forward + backward
    if np.any(denominator <= 0):
        raise ValueError("Nonpositive denominator after background subtraction")
    asymmetry = (alpha * forward - backward) / denominator
    derivative_forward = 2 * alpha * backward / denominator**2
    derivative_backward = -2 * alpha * forward / denominator**2
    variance = (
        derivative_forward**2 * forward_variance
        + derivative_backward**2 * backward_variance
    )
    if np.any(~np.isfinite(variance)) or np.any(variance <= 0):
        raise ValueError("Nonpositive asymmetry variance")
    return asymmetry, np.sqrt(variance)


def make_asymmetry(data: HistogramRun, run: RunConfig, alpha: float = 1.0) -> Spectrum:
    """Build a musrfit type-2 forward/backward asymmetry spectrum."""
    if run.fittype != 2:
        raise ValueError(f"Asymmetry preparation requires fittype 2, got {run.fittype}")
    forward = detector_group(data, run.tokens("forward"))
    backward = detector_group(data, run.tokens("backward"))
    if set(forward) & set(backward):
        raise ValueError("Forward and backward detector groups must not overlap")
    forward_t0, backward_t0 = _t0_groups(data, run, forward, backward)
    forward_t0_int = _require_integer_t0(forward_t0)
    backward_t0_int = _require_integer_t0(backward_t0)

    lower, upper = _common_offset_bounds(
        data, ((forward, forward_t0), (backward, backward_t0)))
    data_range = [int(value) for value in _numbers(run, "data", default=[])]
    if data_range:
        if len(data_range) < 4:
            raise ValueError("Asymmetry data range needs forward-first/last and backward-first/last")
        lower = max(lower, data_range[0] - forward_t0_int[0], data_range[2] - backward_t0_int[0])
        upper = min(upper, data_range[1] - forward_t0_int[0], data_range[3] - backward_t0_int[0])
    else:
        forward_good = _range_from_metadata(data, forward[0])
        backward_good = _range_from_metadata(data, backward[0])
        if forward_good and backward_good:
            lower = max(lower, forward_good[0] - forward_t0_int[0], backward_good[0] - backward_t0_int[0])
            upper = min(upper, forward_good[1] - forward_t0_int[0], backward_good[1] - backward_t0_int[0])
    if upper < lower:
        raise ValueError("Empty overlap after applying t0 and data ranges")
    offsets = np.arange(lower, upper + 1, dtype=int)
    raw_forward = _aligned_group(data, forward, forward_t0, forward_t0_int[0] + offsets)
    raw_backward = _aligned_group(data, backward, backward_t0, backward_t0_int[0] + offsets)

    fixed_background = _numbers(run, "backgr.fix", default=[])
    if fixed_background:
        if len(fixed_background) < 2:
            raise ValueError("backgr.fix needs forward and backward values")
        background_forward, background_backward = fixed_background[:2]
        variance_forward = variance_backward = 0.0
    else:
        ranges = [int(value) for value in _numbers(run, "background", 4)]
        background_forward, variance_forward = _background(
            data, forward, forward_t0, ranges[0], ranges[1])
        background_backward, variance_backward = _background(
            data, backward, backward_t0, ranges[2], ranges[3])

    packing = int(_numbers(run, "packing", default=[1])[0])
    packed_forward, count = _pack(raw_forward, packing)
    packed_backward, _ = _pack(raw_backward, packing)
    corrected_forward = packed_forward - packing * background_forward
    corrected_backward = packed_backward - packing * background_backward
    forward_variance = packed_forward + packing**2 * variance_forward
    backward_variance = packed_backward + packing**2 * variance_backward
    time = offsets[:count].reshape(-1, packing).mean(axis=1) * data.channel_width_us
    metadata = {
        **data.metadata,
        "derived": "asymmetry", "fit_type": 2, "alpha": alpha, "packing": packing,
        "forward": [data.detectors[index].number for index in forward],
        "backward": [data.detectors[index].number for index in backward],
        "t0": [value for pair in zip(forward_t0, backward_t0) for value in pair],
        "background": [background_forward, background_backward],
        "asymmetry_components": {
            "forward": corrected_forward,
            "backward": corrected_backward,
            "forward_variance": forward_variance,
            "backward_variance": backward_variance,
        },
    }
    placeholder = Spectrum(
        time,
        np.zeros_like(time, dtype=float),
        np.ones_like(time, dtype=float),
        metadata,
    )
    asymmetry, error = asymmetry_from_components(placeholder, alpha)
    placeholder.asymmetry = asymmetry
    placeholder.error = error
    return placeholder


def make_histogram(data: HistogramRun, run: RunConfig) -> CountSpectrum:
    """Build packed raw counts for a musrfit type-0 single-histogram fit."""
    if run.fittype != 0:
        raise ValueError(f"Histogram preparation requires fittype 0, got {run.fittype}")
    forward = detector_group(data, run.tokens("forward"))
    forward_t0, _ = _t0_groups(data, run, forward)
    t0 = _require_integer_t0(forward_t0)
    lower, upper = _common_offset_bounds(data, ((forward, forward_t0),))
    data_range = [int(value) for value in _numbers(run, "data", default=[])]
    if data_range:
        if len(data_range) < 2:
            raise ValueError("Single-histogram data range needs first and last channel")
        lower = max(lower, data_range[0] - t0[0])
        upper = min(upper, data_range[1] - t0[0])
    else:
        good = _range_from_metadata(data, forward[0])
        if good:
            lower = max(lower, good[0] - t0[0])
            upper = min(upper, good[1] - t0[0])
    if upper < lower:
        raise ValueError("Empty histogram range after applying t0 and data range")
    offsets = np.arange(lower, upper + 1, dtype=int)
    raw = _aligned_group(data, forward, forward_t0, t0[0] + offsets)
    packing = int(_numbers(run, "packing", default=[1])[0])
    packed, count = _pack(raw, packing)

    fixed_background = _numbers(run, "backgr.fix", default=[])
    if fixed_background:
        background = fixed_background[0] * packing
    else:
        ranges = [int(value) for value in _numbers(run, "background", default=[])]
        background = (
            _background(data, forward, forward_t0, ranges[0], ranges[1])[0] * packing
            if len(ranges) >= 2 else 0.0
        )
    time = offsets[:count].reshape(-1, packing).mean(axis=1) * data.channel_width_us
    metadata = {
        **data.metadata,
        "derived": "histogram", "fit_type": 0, "packing": packing,
        "forward": [data.detectors[index].number for index in forward],
        "t0": forward_t0,
        "background_per_packed_bin": background,
    }
    return CountSpectrum(time, packed, np.sqrt(np.maximum(packed, 1.0)), metadata)


def prepare_run_data(
    data: HistogramRun,
    run: RunConfig,
    *,
    alpha: float = 1.0,
) -> Spectrum | CountSpectrum:
    if run.fittype == 2:
        return make_asymmetry(data, run, alpha)
    if run.fittype == 0:
        return make_histogram(data, run)
    raise NotImplementedError(
        f"fittype {run.fittype} is not implemented; supported data fits are 0 and 2")

