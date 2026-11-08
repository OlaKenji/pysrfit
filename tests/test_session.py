from pathlib import Path

import numpy as np

from pysrfit.io.base import DetectorInfo, HistogramRun, Spectrum
from pysrfit.models import DEFAULT_PACKING, Project, RunConfig
from pysrfit.session import AnalysisSession


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


def test_loaded_header_metadata_is_kept_without_runtime_arrays(tmp_path):
    source = tmp_path / "run.dat"
    source.write_text("data")

    def fake_load(_path):
        return Spectrum(
            [0, 1], [.2, .1], [.01, .01],
            {
                "source": str(source),
                "reader": "test",
                "temperature": "5 K",
                "run_number": np.int64(42),
                "environment": {"pressure": 1.5},
                "asymmetry_components": {"forward": np.ones(2)},
            },
        )

    session = AnalysisSession(Project(runs=[RunConfig()]), loader=fake_load)
    session.attach_data_source(source)
    session.ensure_data_loaded()

    metadata = session.project.runs[0].file_metadata
    assert metadata == {
        "temperature": "5 K",
        "run_number": 42,
        "environment": {"pressure": 1.5},
    }


def test_raw_histogram_default_configuration_uses_detector_metadata(tmp_path):
    time = np.arange(20) * 0.01
    session = AnalysisSession()
    session.raw_data = HistogramRun(
        time,
        np.ones((2, 20)),
        metadata={"temperature": "5 K", "sample": "reference"},
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
    assert run.file_metadata == {"temperature": "5 K", "sample": "reference"}


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


def test_multiple_run_sources_have_independent_caches_and_configuration(tmp_path):
    first = tmp_path / "run1.dat"
    second = tmp_path / "run2.dat"
    first.write_text("one")
    second.write_text("two")
    calls = []

    def fake_load(path):
        calls.append(Path(path))
        amplitude = 0.1 if Path(path) == first else 0.2
        return Spectrum([0, 1], [amplitude, amplitude], [.01, .01])

    session = AnalysisSession(loader=fake_load)
    session.attach_data_source(first)
    session.current_run.options["map"] = ["1"]
    session.ensure_data_loaded()
    session.attach_data_source(second, new_run=True)
    session.current_run.options["map"] = ["2"]
    session.ensure_data_loaded()

    assert session.run_count == 2
    assert len(session.prepared_runs()) == 2
    assert session.project.runs[0].tokens("map") == ["1"]
    assert session.project.runs[1].tokens("map") == ["2"]
    session.project.runs[0].plot_enabled = False
    assert len(session.prepared_runs()) == 2
    assert [index for index, _run, _data in session.prepared_runs(plotted_only=True)] == [1]
    session.select_run(0)
    assert np.allclose(session.data.values, 0.1)
    session.select_run(1)
    assert np.allclose(session.data.values, 0.2)
    assert calls == [first, second]


def test_asymmetry_run_gets_a_floating_alpha_parameter():
    time = np.arange(50) * 0.01
    counts = np.vstack((np.full(50, 110.0), np.full(50, 90.0)))
    counts[:, :5] = 10.0
    raw = HistogramRun(
        time,
        counts,
        detectors=[DetectorInfo(1, "Forw", 5), DetectorInfo(2, "Back", 5)],
    )
    run = RunConfig(fittype=2, options={
        "forward": ["1"], "backward": ["2"], "t0": ["5", "5"],
        "background": ["0", "4", "0", "4"],
        "data": ["5", "30", "5", "30"], "packing": ["1"],
    })
    session = AnalysisSession(Project(parameters=[], runs=[run]))
    session.raw_data = raw
    added = session.process_raw_data()
    assert added == (1,)
    assert run.tokens("alpha") == ["1"]
    alpha = session.project.parameters[0]
    assert alpha.name == "alpha_run1"
    assert not alpha.fixed
    assert alpha.lower > 0
    assert "asymmetry_components" in session.data.metadata


def test_single_histogram_gets_normalization_and_background_parameters():
    time = np.arange(40) * 0.01
    counts = np.full((1, 40), 12.0)
    counts[0, 5:] += 100 * np.exp(-time[:35] / 2.2)
    raw = HistogramRun(
        time,
        counts,
        detectors=[DetectorInfo(1, "Forw", 5, 5, 35)],
    )
    run = RunConfig(fittype=0, options={
        "forward": ["1"], "t0": ["5"], "data": ["5", "35"],
        "packing": ["1"],
    })
    session = AnalysisSession(Project(parameters=[], runs=[run]))
    session.raw_data = raw

    added = session.process_raw_data()

    assert added == (1, 2)
    assert run.tokens("norm") == ["1"]
    assert run.tokens("backgr.fit") == ["2"]
    assert [parameter.name for parameter in session.project.parameters] == [
        "N0_run1", "background_run1"]
    assert all(parameter.lower == 0 for parameter in session.project.parameters)


def test_view_packing_does_not_change_fit_data_or_run_configuration():
    time = np.arange(40) * 0.01
    counts = np.full((1, 40), 12.0)
    counts[0, 5:] += 100 * np.exp(-time[:35] / 2.2)
    raw = HistogramRun(
        time,
        counts,
        detectors=[DetectorInfo(1, "Forw", 5, 5, 35)],
    )
    run = RunConfig(fittype=0, options={
        "forward": ["1"], "t0": ["5"], "data": ["5", "35"],
        "packing": ["1"],
    })
    session = AnalysisSession(Project(parameters=[], runs=[run]))
    session.raw_data = raw
    session.process_raw_data()
    fit_data = session.data

    view_data = session.prepare_view_data(2)

    assert view_data is not fit_data
    assert view_data.metadata["packing"] == 2
    assert len(view_data.time_us) == len(fit_data.time_us) // 2
    assert session.data is fit_data
    assert session.data.metadata["packing"] == 1
    assert run.tokens("packing") == ["1"]


def test_changing_alpha_parameter_refreshes_cached_asymmetry():
    time = np.arange(50) * 0.01
    counts = np.vstack((np.full(50, 110.0), np.full(50, 90.0)))
    counts[:, :5] = 10.0
    raw = HistogramRun(
        time,
        counts,
        detectors=[DetectorInfo(1, "Forw", 5), DetectorInfo(2, "Back", 5)],
    )
    run = RunConfig(fittype=2, options={
        "forward": ["1"], "backward": ["2"], "t0": ["5", "5"],
        "background": ["0", "4", "0", "4"],
        "data": ["5", "30", "5", "30"], "packing": ["1"],
    })
    session = AnalysisSession(Project(parameters=[], runs=[run]))
    session.raw_data = raw
    session.process_raw_data()
    original = session.data.values.copy()

    session.project.parameters[0].value = 2.0
    assert session.refresh_parameter_dependent_data() == 1

    assert np.allclose(original, (100 - 80) / (100 + 80))
    assert np.allclose(session.data.values, (100 - 2 * 80) / (100 + 2 * 80))
    assert session.data.metadata["alpha"] == 2.0

