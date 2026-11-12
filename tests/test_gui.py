"""GUI structure checks that do not require showing a window."""
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt5")
pytest.importorskip("matplotlib")

from PyQt5.QtGui import QKeySequence, QTextCursor
from PyQt5.QtWidgets import QApplication

from pysrfit.gui.main_window import DATA_COLORS, MainWindow
from pysrfit.io.base import DetectorInfo, HistogramRun, Spectrum
from pysrfit.models import RunConfig


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def demo_spectrum(tmp_path_factory):
    path = tmp_path_factory.mktemp("gui-data") / "demo_spectrum.csv"
    time = np.linspace(0.0, 8.0, 401)
    values = 0.2 * np.exp(-0.15 * time) * np.cos(2.0 * np.pi * 1.5 * time)
    np.savetxt(path, np.column_stack((time, values, np.full_like(time, 0.01))),
               delimiter=",")
    return path


def test_analyses_have_independent_projects_and_data(application, demo_spectrum):
    window = MainWindow()
    first = window.current_analysis()
    second = window.add_analysis(data_path=demo_spectrum)

    assert window.analysis_tabs.count() == 2
    assert first.project is not second.project
    assert first.data is None
    assert second.data is not None
    assert window.analysis_tabs.tabText(window.analysis_tabs.currentIndex()) == "demo_spectrum"
    window.close()


def test_data_source_shortcuts(application):
    window = MainWindow()
    actions = {
        action.text(): action
        for menu_action in window.menuBar().actions()
        for action in menu_action.menu().actions()
    }

    assert actions["Add data files…"].shortcut().toString(
        QKeySequence.PortableText) == "Ctrl+D"
    assert actions["Change data source..."].shortcut().toString(
        QKeySequence.PortableText) == "Ctrl+Shift+O"
    file_actions = {
        action.text()
        for menu_action in window.menuBar().actions()
        if menu_action.text() == "&File"
        for action in menu_action.menu().actions()
    }
    assert "Change data source..." in file_actions
    window.close()


def test_theory_picker_and_correction_area(application):
    window = MainWindow()
    analysis = window.current_analysis()
    index = analysis.theory_function_combo.findData("tfieldcos")
    assert index >= 0

    analysis.theory_function_combo.setCurrentIndex(index)
    analysis.theory_edit.moveCursor(QTextCursor.End)
    analysis._insert_theory_function()

    assert analysis.theory_edit.toPlainText().endswith("TFieldCos 1 2")
    assert not analysis.find_t0_button.isEnabled()
    assert not analysis.apply_t0_button.isEnabled()
    assert not analysis.reset_corrections_button.isEnabled()
    assert analysis.fit_type_combo.count() == 2
    assert analysis.fit_type_combo.currentData() == 2
    assert not analysis.processing_rows["alpha"].isHidden()
    assert analysis.processing_rows["norm"].isHidden()
    analysis.fit_type_combo.setCurrentIndex(
        analysis.fit_type_combo.findData(0))
    assert analysis.processing_rows["backward"].isHidden()
    assert analysis.processing_rows["alpha"].isHidden()
    assert not analysis.processing_rows["norm"].isHidden()
    assert not analysis.processing_rows["backgr.fit"].isHidden()
    assert not analysis.processing_rows["lifetime"].isHidden()
    assert analysis.processing_labels["forward"].text() == "Histogram detector(s)"
    assert analysis.data_label.isHidden()
    analysis.run_details_check.setChecked(True)
    assert not analysis.data_label.isHidden()
    assert analysis.tabs.tabText(3) == "Fourier"
    assert analysis.tabs.tabText(5) == "Original .msr"
    assert analysis.tabs.tabText(6) == "Metadata"
    window.close()


def test_raw_histogram_creates_default_run_configuration(application, tmp_path):
    window = MainWindow()
    analysis = window.current_analysis()
    time = np.arange(20) * 0.01
    analysis.session.raw_data = HistogramRun(
        time,
        np.ones((2, 20)),
        detectors=[
            DetectorInfo(1, "Forw", 5, 6, 19),
            DetectorInfo(2, "Back", 7, 8, 19),
        ],
    )
    run = analysis._default_run_for_raw_data(tmp_path / "run.root")
    assert run.fittype == 2
    assert run.tokens("forward") == ["1"]
    assert run.tokens("backward") == ["2"]
    assert run.tokens("t0") == ["5", "7"]
    window.close()


def test_changed_data_source_keeps_configuration_and_reloads_on_change(
        application, tmp_path, monkeypatch):
    source = tmp_path / "run.dat"
    source.write_text("first")
    calls = []

    def fake_load(path):
        calls.append(Path(path))
        return Spectrum([0, 1], [.2, .1], [.01, .01], {"reader": "test"})

    window = MainWindow()
    analysis = window.current_analysis()
    analysis.session._loader = fake_load
    configured_run = RunConfig(
        fittype=2,
        options={"forward": ["1"], "backward": ["2"], "t0": ["5", "7"]},
    )
    analysis.project.runs = [configured_run]

    analysis._attach_data_source(source)
    analysis._ensure_data_loaded()
    analysis._ensure_data_loaded()
    assert len(calls) == 1
    assert analysis.project.runs[0] is configured_run
    assert analysis.project.data_override == str(source.resolve())

    source.write_text("second version with a different size")
    analysis._ensure_data_loaded()
    assert len(calls) == 2
    assert analysis.project.runs[0] is configured_run
    window.close()


def test_run_number_control_changes_source_without_replacing_configuration(
        application, tmp_path):
    first = tmp_path / "gps_2025_003092.dat"
    second = tmp_path / "gps_2025_003093.dat"
    later = tmp_path / "gps_2025_003095.dat"
    first.write_text("first")
    second.write_text("second")
    later.write_text("later")

    def fake_load(path):
        run_number = int(Path(path).stem.rsplit("_", 1)[1])
        return Spectrum(
            [0, 1], [.2, .1], [.01, .01],
            {"reader": "test", "run_number": run_number},
        )

    window = MainWindow()
    analysis = window.current_analysis()
    analysis.session._loader = fake_load
    analysis._attach_data_source(first)
    analysis._ensure_data_loaded()
    configured_run = analysis.project.runs[0]
    configured_run.options["map"] = ["2"]
    analysis._refresh_processing_controls()
    assert analysis.run_number_edit.text() == "3092"

    analysis.run_number_edit.setText("3093")
    analysis.load_run_number()

    assert analysis.project.runs[0] is configured_run
    assert configured_run.tokens("map") == ["2"]
    assert Path(configured_run.source) == second.resolve()
    assert analysis.session.source_path == second.resolve()
    assert configured_run.file_metadata["run_number"] == 3093
    assert analysis.run_number_edit.text() == "3093"

    analysis.shift_run_number(1)

    assert Path(configured_run.source) == later.resolve()
    assert configured_run.file_metadata["run_number"] == 3095
    assert analysis.run_number_edit.text() == "3095"
    window.close()


def test_multiple_runs_have_independent_maps_and_plot_selection(
        application, demo_spectrum):
    window = MainWindow()
    analysis = window.add_analysis(data_paths=[
        demo_spectrum,
        demo_spectrum,
    ])
    assert analysis.session.run_count == 2
    assert len(analysis.session.prepared_runs()) == 2

    analysis.run_combo.setCurrentIndex(0)
    analysis.mapping_edit.setText("1")
    analysis._apply_processing_controls()
    analysis.run_combo.setCurrentIndex(1)
    analysis.mapping_edit.setText("2")
    analysis._apply_processing_controls()
    assert analysis.project.runs[0].tokens("map") == ["1"]
    assert analysis.project.runs[1].tokens("map") == ["2"]

    analysis.mapping_edit.clear()
    analysis._apply_processing_controls()
    analysis.run_plot_check.setChecked(False)
    analysis.redraw()
    assert len(analysis.canvas.figure.axes) == 1
    window.close()


def test_apply_parameters_refreshes_alpha_spectrum(application):
    window = MainWindow()
    analysis = window.current_analysis()
    time = np.arange(50) * 0.01
    counts = np.vstack((np.full(50, 110.0), np.full(50, 90.0)))
    counts[:, :5] = 10.0
    run = RunConfig(fittype=2, options={
        "forward": ["1"], "backward": ["2"], "t0": ["5", "5"],
        "background": ["0", "4", "0", "4"],
        "data": ["5", "30", "5", "30"], "packing": ["1"],
    })
    analysis.project.runs = [run]
    analysis.session.raw_data = HistogramRun(
        time,
        counts,
        detectors=[DetectorInfo(1, "Forw", 5), DetectorInfo(2, "Back", 5)],
    )
    analysis.session.process_raw_data()
    analysis._populate_parameters_only()
    alpha_index = int(run.tokens("alpha")[0])
    changed_lines = []
    for line in analysis.parameters_edit.toPlainText().splitlines():
        body = line.split("#", 1)[0].split()
        if body and body[0] == str(alpha_index):
            body[2] = "2"
            line = " ".join(body)
        changed_lines.append(line)
    analysis.parameters_edit.setPlainText("\n".join(changed_lines))

    analysis.apply_parameters()

    assert np.allclose(
        analysis.data.values,
        (2 * 100 - 80) / (2 * 100 + 80),
    )
    assert analysis.data.metadata["alpha"] == 2.0
    window.close()


def test_switching_to_single_histogram_configures_count_parameters(application):
    window = MainWindow()
    analysis = window.current_analysis()
    time = np.arange(50) * 0.01
    counts = np.vstack((np.full(50, 110.0), np.full(50, 90.0)))
    counts[:, :5] = 10.0
    run = RunConfig(fittype=2, options={
        "forward": ["1"], "backward": ["2"], "t0": ["5", "5"],
        "background": ["0", "4", "0", "4"],
        "data": ["5", "30", "5", "30"], "packing": ["1"],
    })
    analysis.project.runs = [run]
    analysis.session.raw_data = HistogramRun(
        time,
        counts,
        detectors=[DetectorInfo(1, "Forw", 5), DetectorInfo(2, "Back", 5)],
    )
    analysis._refresh_processing_controls()
    analysis.fit_type_combo.setCurrentIndex(
        analysis.fit_type_combo.findData(0))

    analysis._apply_processing_controls()

    assert run.fittype == 0
    assert not run.tokens("backward")
    assert not run.tokens("alpha")
    assert not run.tokens("background")
    assert run.tokens("t0") == ["5"]
    assert run.tokens("data") == ["5", "30"]
    assert run.tokens("norm")
    assert run.tokens("backgr.fit")
    assert analysis.norm_edit.text() == run.tokens("norm")[0]
    assert analysis.background_fit_edit.text() == run.tokens("backgr.fit")[0]
    assert analysis.data.kind == "histogram"
    window.close()


def test_revert_fit_toggles_exact_pre_fit_text_and_latest_result(
        application, monkeypatch, demo_spectrum):
    window = MainWindow()
    analysis = window.add_analysis(data_path=demo_spectrum)
    before_text = analysis.parameters_edit.toPlainText() + "\n\n# Keep this layout"
    analysis.parameters_edit.setPlainText(before_text)
    before_values = {
        parameter.index: parameter.value for parameter in analysis.project.parameters
    }
    fitted_values = {
        index: value + index for index, value in before_values.items()
    }

    def fake_fit(_project, datasets):
        assert datasets
        return SimpleNamespace(
            valid=True,
            values=fitted_values,
            errors={index: 0.001 * index for index in fitted_values},
            chi2=12.0,
            ndf=10,
            run_chi2=[12.0],
            run_points=[20],
            automatically_fixed=[],
        )

    monkeypatch.setattr("pysrfit.gui.main_window.fit_global", fake_fit)
    saves = []
    monkeypatch.setattr(
        analysis, "save_project", lambda: saves.append(True) or True)
    analysis.fit()

    result_text = analysis.parameters_edit.toPlainText()
    assert saves == [True]
    assert result_text != before_text
    assert analysis.revert_fit_button.isEnabled()
    assert analysis.revert_fit_button.text() == "Revert fit"

    other_analysis = window.add_analysis()
    assert analysis.revert_fit_button.isEnabled()
    assert not other_analysis.revert_fit_button.isEnabled()

    analysis.revert_fit_button.click()
    assert analysis.parameters_edit.toPlainText() == before_text
    assert analysis.project.parameter_values() == before_values
    assert analysis.revert_fit_button.text() == "Restore fit"

    analysis.revert_fit_button.click()
    assert analysis.parameters_edit.toPlainText() == result_text
    assert analysis.project.parameter_values() == fitted_values
    assert analysis.revert_fit_button.text() == "Revert fit"
    window.close()


def test_save_uses_current_path_and_save_as_chooses_a_new_path(
        application, tmp_path, monkeypatch):
    window = MainWindow()
    analysis = window.current_analysis()
    current = tmp_path / "current.pysrfit"
    analysis.project.location = current

    assert analysis.save_project()
    assert current.is_file()

    replacement = tmp_path / "renamed"
    monkeypatch.setattr(
        "pysrfit.gui.main_window.QFileDialog.getSaveFileName",
        lambda *_args, **_kwargs: (str(replacement), ""),
    )
    assert analysis.save_as_dialog()
    assert (tmp_path / "renamed.pysrfit").is_file()
    assert analysis.project.location == tmp_path / "renamed.pysrfit"
    window.close()


def test_fourier_plot_shows_data_markers_and_fit_line(application, demo_spectrum):
    window = MainWindow()
    analysis = window.add_analysis(data_path=demo_spectrum)
    analysis.project.runs[0].file_metadata["run_number"] = 42
    analysis.mode.setCurrentIndex(1)
    analysis.redraw()

    axis = analysis.canvas.figure.axes[0]
    artists = {artist.get_label(): artist for artist in axis.lines}
    assert artists["42"].get_marker() == "o"
    assert artists["42"].get_markersize() == 7
    assert artists["42"].get_markeredgecolor() == "black"
    assert artists["42"].get_linestyle() == "None"
    assert artists["_nolegend_"].get_linestyle() == "-"
    assert artists["42"].get_color() == artists["_nolegend_"].get_color()
    assert [text.get_text() for text in axis.get_legend().get_texts()] == ["42"]
    assert axis.get_title() == ""
    assert not any(line.get_visible() for line in axis.get_xgridlines())
    window.close()


def test_time_plot_uses_distinct_run_markers_and_data_only_legend(
        application, demo_spectrum):
    window = MainWindow()
    analysis = window.add_analysis(data_paths=[
        demo_spectrum,
        demo_spectrum,
        demo_spectrum,
    ])
    analysis.project.runs[0].file_metadata["run_number"] = 41
    analysis.project.runs[1].file_metadata["run_number"] = 42
    analysis.project.runs[2].file_metadata["run_number"] = 43
    analysis.redraw()

    axis = analysis.canvas.figure.axes[0]
    data_artists = {
        container.get_label(): container.lines[0]
        for container in axis.containers
        if container.get_label() in {"41", "42", "43"}
    }
    assert data_artists["41"].get_marker() == "o"
    assert data_artists["42"].get_marker() == "s"
    assert data_artists["43"].get_marker() == "^"
    assert data_artists["41"].get_color() == DATA_COLORS[0]
    assert data_artists["42"].get_color() == DATA_COLORS[1]
    assert data_artists["43"].get_color() == DATA_COLORS[2]
    assert [text.get_text() for text in axis.get_legend().get_texts()] == [
        "41", "42", "43"]
    assert all(
        artist.get_label() == "_nolegend_"
        for artist in axis.lines
        if artist.get_linestyle() == "-"
    )
    assert axis.get_title() == ""
    assert not any(line.get_visible() for line in axis.get_ygridlines())
    window.close()


def test_run_metadata_table_separates_file_and_user_values(application, tmp_path):
    source = tmp_path / "run.dat"
    source.write_text("data")

    def fake_load(_path):
        return Spectrum(
            [0, 1], [.2, .1], [.01, .01],
            {"reader": "test", "temperature": "5 K", "run_number": 42},
        )

    window = MainWindow()
    analysis = window.current_analysis()
    analysis.session._loader = fake_load
    analysis._attach_data_source(source)
    analysis._ensure_data_loaded()
    analysis.project.runs[0].user_metadata["comment"] = "field cooled"
    analysis._refresh_metadata_table()

    rows = {
        analysis.metadata_table.item(row, 0).text(): (
            analysis.metadata_table.item(row, 1).text(),
            analysis.metadata_table.item(row, 2).text(),
        )
        for row in range(analysis.metadata_table.rowCount())
    }
    assert rows["temperature"] == ("5 K", "Data file")
    assert rows["run_number"] == ("42", "Data file")
    assert rows["comment"] == ("field cooled", "User")
    window.close()
