"""GUI structure checks that do not require showing a window."""
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt5")
pytest.importorskip("matplotlib")

from PyQt5.QtGui import QTextCursor
from PyQt5.QtWidgets import QApplication

from mysr.gui.main_window import MainWindow
from mysr.io.base import DetectorInfo, HistogramRun, Spectrum
from mysr.models import RunConfig


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


def test_analyses_have_independent_projects_and_data(application):
    window = MainWindow()
    first = window.current_analysis()
    second = window.add_analysis(data_path=ROOT / "examples" / "demo_spectrum.csv")

    assert window.analysis_tabs.count() == 2
    assert first.project is not second.project
    assert first.data is None
    assert second.data is not None
    assert window.analysis_tabs.tabText(window.analysis_tabs.currentIndex()) == "demo_spectrum"
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
