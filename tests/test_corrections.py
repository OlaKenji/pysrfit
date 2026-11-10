import numpy as np
import pytest

from pysrfit.corrections import RunCorrectionState
from pysrfit.io.base import DetectorInfo, HistogramRun
from pysrfit.models import RunConfig


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

    assert state.marker_values(1)["t0"] == 9
    assert state.marker_values(1)["data"] == [10, 24]
    assert state.marker_values(0)["background"] == [2, 6]
    assert run.tokens("t0") == ["9"]
    assert run.tokens("data") == ["10", "24", "10", "24"]
    assert run.tokens("background") == ["2", "6", "2", "6"]


def test_one_t0_value_populates_and_stays_compact():
    run = RunConfig(fittype=2, options={
        "forward": ["3"], "backward": ["4"], "t0": ["6"],
        "data": ["6", "25", "8", "27"],
        "background": ["0", "3", "1", "4"],
    })
    state = RunCorrectionState(example_histograms(), run)

    assert state.marker_values(0)["t0"] == 6
    assert state.marker_values(1)["t0"] == 6

    state.apply_to(run)
    assert run.tokens("t0") == ["6"]


def test_current_histogram_scope_preserves_forward_and_backward_settings():
    run = RunConfig(fittype=2, options={
        "forward": ["3"], "backward": ["4"],
        "t0": ["5", "7"], "data": ["6", "25", "8", "27"],
        "background": ["0", "3", "1", "4"],
    })
    state = RunCorrectionState(example_histograms(), run)

    state.set_marker(0, "t", 9, apply_all=False)
    state.set_marker(0, "d", 10, apply_all=False)
    state.set_marker(1, "B", 6, apply_all=False)
    state.apply_to(run)

    assert state.marker_values(0)["t0"] == 9
    assert state.marker_values(1)["t0"] == 7
    assert run.tokens("t0") == ["9", "7"]
    assert run.tokens("data") == ["10", "25", "8", "27"]
    assert run.tokens("background") == ["0", "3", "1", "6"]


def test_all_histograms_scope_keeps_latest_setting():
    run = RunConfig(fittype=2, options={
        "forward": ["3"], "backward": ["4"],
        "t0": ["5", "7"], "data": ["6", "25", "8", "27"],
        "background": ["0", "3", "1", "4"],
    })
    state = RunCorrectionState(example_histograms(), run)

    state.set_marker(0, "t", 8, apply_all=True)
    state.set_marker(1, "t", 9, apply_all=True)
    state.set_marker(0, "d", 10, apply_all=True)

    assert state.marker_values(0)["t0"] == 9
    assert state.marker_values(1)["t0"] == 9
    assert state.marker_values(0)["data"][0] == 10
    assert state.marker_values(1)["data"][0] == 10


def test_large_detector_banks_default_to_broadcast_scope():
    detector_count = 20
    histograms = HistogramRun(
        np.arange(30, dtype=float) * 0.01,
        np.ones((detector_count, 30)),
        detectors=[
            DetectorInfo(number, f"Detector {number}", 5, 6, 25)
            for number in range(1, detector_count + 1)
        ],
    )
    run = RunConfig(fittype=2, options={
        "forward": ["1-10"], "backward": ["11-20"],
        "t0": ["5"], "data": ["6", "25", "6", "25"],
        "background": ["0", "3", "0", "3"],
    })

    assert not RunCorrectionState(example_histograms(), RunConfig(
        fittype=2,
        options={"forward": ["3"], "backward": ["4"]},
    )).default_apply_all
    assert RunCorrectionState(histograms, run).default_apply_all


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


def test_single_histogram_fitted_background_disables_background_markers():
    run = RunConfig(fittype=0, options={
        "forward": ["3"], "t0": ["5"], "data": ["6", "25"],
        "background": ["0", "3"], "backgr.fit": ["2"],
    })
    state = RunCorrectionState(example_histograms(), run)
    assert not state.background_editable
    with pytest.raises(ValueError, match="ignored"):
        state.set_marker(0, "b", 1)
    state.apply_to(run)
    assert not run.tokens("background")
