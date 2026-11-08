import numpy as np
import pytest
from pysrfit.models import Project, Parameter, RunConfig
from pysrfit.io.asymmetry import asymmetry_from_components
from pysrfit.io.base import CountSpectrum, Spectrum
from pysrfit.core.fit import (
    _fit_range,
    evaluate_data_model,
    fit_global,
    transform_asymmetry_model_for_alpha,
    used_parameter_indices,
)


def test_minuit_recovers_simple_model():
    pytest.importorskip("iminuit")
    t = np.linspace(0, 8, 101)
    y = .24 * np.exp(-.2 * t)
    d = Spectrum(t, y, np.full_like(t, .01))
    p = Project(parameters=[Parameter(1, "A", .2, .01), Parameter(2, "lam", .3, .01)],
                theory="asymmetry 1\nsimplExpo 2")
    result = fit_global(p, [(RunConfig(), d)])
    assert result.valid
    assert result.values[1] == pytest.approx(.24, abs=1e-4)
    assert result.values[2] == pytest.approx(.2, abs=1e-4)


def test_parameter_usage_follows_theory_functions_maps_and_run_settings():
    project = Project(
        parameters=[Parameter(index, f"p{index}", 1.0) for index in range(1, 7)],
        theory="asymmetry 1\nsimplExpo fun1",
        functions_text="fun1 = par2 + map1\nfun2 = par5",
    )
    run = RunConfig(options={"map": ["3"], "alpha": ["4"]})
    data = Spectrum(
        [0.0, 1.0],
        [0.0, 0.0],
        [1.0, 1.0],
        {"asymmetry_components": {}},
    )

    assert used_parameter_indices(project, [(run, data)]) == {1, 2, 3, 4}


def test_parameter_usage_includes_single_histogram_references():
    project = Project(
        parameters=[Parameter(index, f"p{index}", 1.0) for index in range(1, 6)],
        theory="asymmetry 1",
    )
    run = RunConfig(fittype=0, options={
        "norm": ["2"], "backgr.fit": ["3"], "lifetime": ["4"],
    })
    data = CountSpectrum([0.0, 1.0], [10.0, 8.0], [3.0, 3.0])

    assert used_parameter_indices(project, [(run, data)]) == {1, 2, 3, 4}


def test_alpha_transforms_theory_to_fixed_reference_asymmetry():
    forward = np.array([100.0, 120.0])
    backward = np.array([80.0, 70.0])
    alpha = 1.3
    corrected = (alpha * forward - backward) / (alpha * forward + backward)
    reference = (forward - backward) / (forward + backward)

    assert np.allclose(
        transform_asymmetry_model_for_alpha(corrected, alpha),
        reference,
    )
    with pytest.raises(ValueError, match="positive"):
        transform_asymmetry_model_for_alpha(corrected, 0.0)


def test_minuit_automatically_fixes_unused_parameters():
    pytest.importorskip("iminuit")
    t = np.linspace(0, 8, 101)
    y = .24 * np.exp(-.2 * t)
    data = Spectrum(t, y, np.full_like(t, .01))
    project = Project(
        parameters=[
            Parameter(1, "A", .2, .01),
            Parameter(2, "lam", .3, .01),
            Parameter(3, "unused", 123.0, 1.0),
        ],
        theory="asymmetry 1\nsimplExpo 2",
    )

    result = fit_global(project, [(RunConfig(), data)])

    assert result.valid
    assert result.automatically_fixed == [3]
    assert result.values[3] == 123.0
    assert result.errors[3] == 0.0
    assert result.ndf == len(t) - 2


def test_musrfit_single_histogram_model():
    time = np.array([0.0, 1.0, 2.0])
    data = CountSpectrum(time, np.ones(3), np.ones(3), {"background_per_packed_bin": 3.0})
    project = Project(
        parameters=[
            Parameter(1, "A", 0.2),
            Parameter(2, "N0", 100.0),
            Parameter(3, "background", 4.0),
            Parameter(4, "lifetime", 2.0),
        ],
        theory="asymmetry 1",
        runs=[RunConfig(fittype=0, options={
            "norm": ["2"], "backgr.fit": ["3"], "lifetime": ["4"],
        })],
    )
    model = evaluate_data_model(project, data, time)
    expected = 100 * np.exp(-time / 2.0) * 1.2 + 4
    assert np.allclose(model, expected)


def test_run_mapping_selects_local_parameters():
    time = np.array([0.0, 1.0])
    data = Spectrum(time, np.ones(2), np.ones(2))
    project = Project(
        parameters=[Parameter(1, "run1", 0.2), Parameter(2, "run2", 0.3)],
        theory="asymmetry map1",
    )
    run1 = RunConfig(options={"map": ["1"]})
    run2 = RunConfig(options={"map": ["2"]})
    assert np.allclose(evaluate_data_model(project, data, time, run=run1), 0.2)
    assert np.allclose(evaluate_data_model(project, data, time, run=run2), 0.3)


def test_fit_range_is_independent_from_plot_range():
    project = Project(display={"t_start": 2.0, "t_stop": 3.0})
    assert _fit_range(project, RunConfig()) == (-np.inf, np.inf)
    assert _fit_range(
        project, RunConfig(options={"fit": ["0.5", "7.5"]})) == (0.5, 7.5)


def test_global_fit_recovers_shared_and_run_local_parameters():
    pytest.importorskip("iminuit")
    time = np.linspace(0, 5, 101)
    error = np.full_like(time, 0.01)
    first = Spectrum(time, 0.24 * np.exp(-0.15 * time), error)
    second = Spectrum(time, 0.24 * np.exp(-0.30 * time), error)
    project = Project(
        parameters=[
            Parameter(1, "shared_A", 0.2, 0.01),
            Parameter(2, "lambda_1", 0.1, 0.01),
            Parameter(3, "lambda_2", 0.2, 0.01),
        ],
        theory="asymmetry 1\nsimplExpo map1",
    )
    run1 = RunConfig(label="one", options={"map": ["2"]})
    run2 = RunConfig(label="two", options={"map": ["3"]})
    result = fit_global(project, [(run1, first), (run2, second)])
    assert result.valid
    assert result.values[1] == pytest.approx(0.24, abs=1e-4)
    assert result.values[2] == pytest.approx(0.15, abs=1e-4)
    assert result.values[3] == pytest.approx(0.30, abs=1e-4)
    assert len(result.run_chi2) == 2


def test_fit_recovers_alpha_from_detector_components(monkeypatch):
    pytest.importorskip("iminuit")
    time = np.linspace(0, 5, 101)
    true_alpha = 1.3
    expected_asymmetry = 0.2
    backward = np.full_like(time, 10_000.0)
    forward = backward * (
        (1 + expected_asymmetry) / (true_alpha * (1 - expected_asymmetry)))
    metadata = {"asymmetry_components": {
        "forward": forward,
        "backward": backward,
        "forward_variance": forward,
        "backward_variance": backward,
    }}
    data = Spectrum(time, np.zeros_like(time), np.ones_like(time), metadata)
    data.asymmetry, data.error = asymmetry_from_components(data, 0.9)
    project = Project(
        parameters=[
            Parameter(1, "alpha", 0.9, 0.01, 0.01, 10.0),
            Parameter(2, "A", expected_asymmetry, 0.0),
        ],
        theory="asymmetry 2",
    )
    run = RunConfig(options={"alpha": ["1"]})
    calls = []

    def tracked_reference_asymmetry(spectrum, alpha):
        calls.append(alpha)
        return asymmetry_from_components(spectrum, alpha)

    monkeypatch.setattr(
        "pysrfit.core.fit.asymmetry_from_components",
        tracked_reference_asymmetry,
    )

    result = fit_global(project, [(run, data)])

    assert result.valid
    assert result.values[1] == pytest.approx(true_alpha, abs=1e-4)
    assert calls == [1.0]
