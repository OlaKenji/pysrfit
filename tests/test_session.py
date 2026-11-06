from pathlib import Path

import numpy as np

from mysr.io.base import DetectorInfo, HistogramRun, Spectrum
from mysr.models import DEFAULT_PACKING, Project, RunConfig
from mysr.session import AnalysisSession


def test_changed_source_keeps_configuration_and_reloads_on_change(tmp_path):
    source = tmp_path / "run.dat"
    source.write_text("first")
    calls = []

    def fake_load(path):
        calls.append(Path(path))
        return Spectrum([0, 1], [.2, .1], [.01, .01], {"reader": "test"})

    configured_run = RunConfig(
        fittype=2,
        options={"forward": ["1"], "backward": ["2"], "t0": ["5", "7"]},
    )
    session = AnalysisSession(Project(runs=[configured_run]), loader=fake_load)
    session.attach_data_source(source)

    assert session.ensure_data_loaded().loaded
    assert not session.ensure_data_loaded().loaded
    assert len(calls) == 1
    assert session.project.runs[0] is configured_run
    assert session.project.data_override == str(source.resolve())

    source.write_text("second version with a different size")
    outcome = session.ensure_data_loaded()
    assert outcome.loaded and outcome.reloaded
    assert len(calls) == 2
    assert session.project.runs[0] is configured_run


def test_raw_histogram_default_configuration_uses_detector_metadata(tmp_path):
    time = np.arange(20) * 0.01
    session = AnalysisSession()
    session.raw_data = HistogramRun(
        time,
        np.ones((2, 20)),
        detectors=[
            DetectorInfo(1, "Forw", 5, 6, 19),
            DetectorInfo(2, "Back", 7, 8, 19),
        ],
    )
    run = session.default_run_for_raw_data(tmp_path / "run.root")
    assert run.fittype == 2
    assert run.tokens("forward") == ["1"]
    assert run.tokens("backward") == ["2"]
    assert run.tokens("t0") == ["5", "7"]
    assert run.tokens("packing") == [str(DEFAULT_PACKING)]


def test_project_data_discovery_does_not_load_arrays(tmp_path):
    source = tmp_path / "run.dat"
    source.write_text("data")
    calls = []
    project = Project(data_override=str(source))
    session = AnalysisSession(project, loader=lambda path: calls.append(path))
    assert session.discover_project_data_source() == source.resolve()
    assert calls == []
    assert session.data is None
    assert session.raw_data is None

