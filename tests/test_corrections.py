import numpy as np
import pytest

from mysr.corrections import RunCorrectionState
from mysr.io.base import DetectorInfo, HistogramRun
from mysr.models import RunConfig


def example_histograms():
    return HistogramRun(
        np.arange(30, dtype=float) * 0.01,
        np.vstack((np.arange(30), np.arange(30)[::-1])),
        detectors=[
            DetectorInfo(3, "Forward", 5, 6, 25),
            DetectorInfo(4, "Backward", 7, 8, 27),
        ],
    )


def test_visual_markers_apply_in_musrfit_order():
    run = RunConfig(fittype=2, options={
        "forward": ["3"], "backward": ["4"],
        "t0": ["5", "7"], "data": ["6", "25", "8", "27"],
        "background": ["0", "3", "1", "4"],
    })
    state = RunCorrectionState(example_histograms(), run)
    assert [target.role for target in state.targets] == ["forward", "backward"]
    assert state.marker_values(0)["t0"] == 5
    assert state.marker_values(1)["data"] == [8, 27]

    state.set_marker(0, "t", 9)
    state.set_marker(0, "d", 10)
    state.set_marker(0, "D", 24)
    state.set_marker(1, "b", 2)
    state.set_marker(1, "B", 6)
    state.apply_to(run)

    assert run.tokens("t0") == ["9", "7"]
    assert run.tokens("data") == ["10", "24", "8", "27"]
    assert run.tokens("background") == ["0", "3", "2", "6"]


def test_visual_marker_range_validation():
    run = RunConfig(fittype=2, options={
        "forward": ["3"], "backward": ["4"], "t0": ["5", "7"],
        "data": ["6", "25", "8", "27"], "background": ["0", "3", "1", "4"],
    })
    state = RunCorrectionState(example_histograms(), run)
    state.set_marker(0, "d", 20)
    state.set_marker(0, "D", 10)
    with pytest.raises(ValueError, match="start exceeds end"):
        state.apply_to(run)


def test_asymmetry_visual_editor_requires_background_ranges():
    run = RunConfig(fittype=2, options={
        "forward": ["3"], "backward": ["4"], "t0": ["5", "7"],
        "data": ["6", "25", "8", "27"],
    })
    state = RunCorrectionState(example_histograms(), run)
    with pytest.raises(ValueError, match="background limits"):
        state.apply_to(run)


def test_single_histogram_visual_markers():
    run = RunConfig(fittype=0, options={
        "forward": ["3"], "t0": ["5"],
        "data": ["6", "25"], "background": ["0", "3"],
    })
    state = RunCorrectionState(example_histograms(), run)
    assert len(state.targets) == 1
    state.set_marker(0, "t", 8)
    state.apply_to(run)
    assert run.tokens("t0") == ["8"]
    assert run.tokens("data") == ["6", "25"]
