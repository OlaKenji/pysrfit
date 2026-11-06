import numpy as np
import pytest
from mysr.models import Project, Parameter, RunConfig
from mysr.io.base import CountSpectrum, Spectrum
from mysr.core.fit import evaluate_data_model, fit_project


def test_minuit_recovers_simple_model():
    pytest.importorskip("iminuit")
    t = np.linspace(0, 8, 101)
    y = .24 * np.exp(-.2 * t)
    d = Spectrum(t, y, np.full_like(t, .01))
    p = Project(parameters=[Parameter(1, "A", .2, .01), Parameter(2, "lam", .3, .01)],
                theory="asymmetry 1\nsimplExpo 2")
    result = fit_project(p, d)
    assert result.valid
    assert result.values[1] == pytest.approx(.24, abs=1e-4)
    assert result.values[2] == pytest.approx(.2, abs=1e-4)


def test_musrfit_single_histogram_model():
    time = np.array([0.0, 1.0, 2.0])
    data = CountSpectrum(time, np.ones(3), np.ones(3), {"background_per_packed_bin": 3.0})
    project = Project(
        parameters=[Parameter(1, "A", 0.2), Parameter(2, "N0", 100.0)],
        theory="asymmetry 1",
        runs=[RunConfig(fittype=0, options={"norm": ["2"]})],
    )
    model = evaluate_data_model(project, data, time)
    expected = 100 * np.exp(-time / 2.197019) * 1.2 + 3
    assert np.allclose(model, expected)
