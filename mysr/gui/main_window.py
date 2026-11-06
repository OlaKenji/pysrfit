"""Single-window GUI: msr editor + model preview + data/fit tools."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PyQt5.QtCore import Qt, QSignalBlocker, pyqtSignal
from PyQt5.QtGui import QFontDatabase
from PyQt5.QtWidgets import (QMainWindow, QWidget, QSplitter, QVBoxLayout, QHBoxLayout,
    QTabWidget, QPlainTextEdit, QLabel, QPushButton, QToolBar, QAction,
    QFileDialog, QMessageBox, QComboBox, QCheckBox, QGroupBox,
    QDoubleSpinBox, QLineEdit, QSpinBox)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure

from ..models import DEFAULT_PACKING, Project, RunConfig
from ..msr import read_msr
from ..parameters import (
    DEFAULT_SIGNIFICANT_DIGITS,
    format_parameter_text,
    parse_parameter_text,
    renumber_parameter_text,
)
from ..session import AnalysisSession
from ..core.fit import evaluate_data_model, fit_project
from ..core.theory import evaluate_theory
from ..core.functions import REGISTRY, CUSTOM_EXAMPLE, activate_user_code
from ..io.base import CountSpectrum, HistogramRun, Spectrum
from .run_corrections import RunCorrectionsDialog


class PlotCanvas(FigureCanvas):
    def __init__(self):
        self.figure = Figure(figsize=(8, 6), constrained_layout=True)
        super().__init__(self.figure)


class MainWindow(QMainWindow):
    def __init__(self, path: str | None = None):
        super().__init__()
        self.setWindowTitle("mysr — μSR analysis")
        self.resize(1390, 820)
        self._make_gui()
        self.add_analysis(path=path)

    def _make_gui(self):
        file_menu = self.menuBar().addMenu("&File")
        self._action(file_menu, "New analysis", self.add_analysis, "Ctrl+N")
        self._action(file_menu, "Open project / .msr in new tab...", self.open_dialog, "Ctrl+O")
        self._action(file_menu, "Open data in new analysis...", self.open_data_dialog, "Ctrl+Shift+O")
        file_menu.addSeparator()
        self._action(file_menu, "Save current project...", lambda: self._call("save_dialog"), "Ctrl+S")
        self._action(file_menu, "Export current plot as PNG...", lambda: self._call("export_plot"))
        self._action(file_menu, "Close analysis tab", self.close_current_analysis, "Ctrl+W")
        file_menu.addSeparator()
        self._action(file_menu, "Quit", self.close, "Ctrl+Q")

        analysis_menu = self.menuBar().addMenu("&Analysis")
        self._action(analysis_menu, "Change data source...", lambda: self._call("change_data_source_dialog"), "Ctrl+D")
        self._action(analysis_menu, "Update theory preview", lambda: self._call("redraw"), "Ctrl+R")
        self._action(analysis_menu, "Fit", lambda: self._call("fit"), "Ctrl+F")

        toolbar = QToolBar("Analysis tabs")
        self.addToolBar(toolbar)
        for text, callback in [
            ("New analysis", self.add_analysis),
            ("Open project", self.open_dialog),
            ("New analysis from data", self.open_data_dialog),
            ("Close tab", self.close_current_analysis),
        ]:
            button = QPushButton(text)
            button.clicked.connect(callback)
            toolbar.addWidget(button)

        self.analysis_tabs = QTabWidget()
        self.analysis_tabs.setTabsClosable(True)
        self.analysis_tabs.setMovable(True)
        self.analysis_tabs.setDocumentMode(True)
        self.analysis_tabs.tabCloseRequested.connect(self.close_analysis)
        self.analysis_tabs.currentChanged.connect(self._current_changed)
        self.setCentralWidget(self.analysis_tabs)
        self.statusBar().showMessage("Ready")

    @staticmethod
    def _action(menu, text, callback, shortcut=None):
        action = QAction(text, menu)
        action.triggered.connect(callback)
        if shortcut:
            action.setShortcut(shortcut)
        menu.addAction(action)

    def current_analysis(self):
        widget = self.analysis_tabs.currentWidget()
        return widget if isinstance(widget, AnalysisWorkspace) else None

    def _call(self, method):
        analysis = self.current_analysis()
        if analysis is not None:
            return getattr(analysis, method)()

    def add_analysis(self, checked=False, *, path=None, data_path=None):
        analysis = AnalysisWorkspace(path=path)
        analysis.title_changed.connect(lambda: self._refresh_tab_title(analysis))
        index = self.analysis_tabs.addTab(analysis, analysis.document_title())
        self.analysis_tabs.setCurrentIndex(index)
        if data_path is not None:
            try:
                source = Path(data_path).expanduser().resolve()
                analysis.project.title = source.stem
                analysis._attach_data_source(source)
                analysis._ensure_data_loaded()
                analysis.redraw()
                analysis.title_changed.emit()
            except Exception as exc:
                analysis._error(f"Data import failed: {exc}")
        self._current_changed(index)
        return analysis

    def open_dialog(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Open mysr or musrfit file", "",
            "Projects and musrfit (*.mysr *.msr);;All files (*)")
        if file:
            self.add_analysis(path=file)

    def open_data_dialog(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Open μSR data in a new analysis", "",
            "Data (*.txt *.csv *.dat *.nxs *.nexus *.h5 *.hdf5 *.bin *.mdu *.root *.mud);;All files (*)")
        if file:
            self.add_analysis(data_path=file)

    def close_current_analysis(self):
        index = self.analysis_tabs.currentIndex()
        if index >= 0:
            self.close_analysis(index)

    def close_analysis(self, index):
        widget = self.analysis_tabs.widget(index)
        self.analysis_tabs.removeTab(index)
        if widget is not None:
            widget.deleteLater()
        if self.analysis_tabs.count() == 0:
            self.add_analysis()

    def _refresh_tab_title(self, analysis):
        index = self.analysis_tabs.indexOf(analysis)
        if index >= 0:
            self.analysis_tabs.setTabText(index, analysis.document_title())
        if analysis is self.current_analysis():
            self._current_changed(index)

    def _current_changed(self, _index):
        analysis = self.current_analysis()
        title = analysis.document_title() if analysis is not None else "μSR analysis"
        self.setWindowTitle(f"mysr — {title}")


class AnalysisWorkspace(QWidget):
    title_changed = pyqtSignal()

    def __init__(self, path: str | None = None):
        super().__init__()
        self.session = AnalysisSession()
        self._make_gui()
        self._populate()
        if path:
            self.open_path(path)
        else:
            self._log("Demo model loaded. Open an .msr file or import a 3-column spectrum.")
            self.redraw()

    @property
    def project(self) -> Project:
        return self.session.project

    @property
    def data(self) -> Spectrum | CountSpectrum | None:
        return self.session.data

    @property
    def raw_data(self) -> HistogramRun | None:
        return self.session.raw_data

    def document_title(self):
        if self.project.location is not None:
            return self.project.location.name
        return self.project.title or "Untitled analysis"

    def _make_gui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        toolbar = QToolBar("Analysis controls")
        for text, fn in [("Fit", self.fit), ("Save", self.save_dialog)]:
            b = QPushButton(text)
            b.clicked.connect(fn)
            toolbar.addWidget(b)
        toolbar.addSeparator()
        self.mode = QComboBox()
        self.mode.addItems(["Time spectrum", "Fourier (exploratory)"])
        self.mode.currentIndexChanged.connect(self.redraw)
        toolbar.addWidget(self.mode)
        self.residual_check = QCheckBox("Show residuals")
        self.residual_check.stateChanged.connect(self.redraw)
        toolbar.addWidget(self.residual_check)
        root.addWidget(toolbar)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, 1)
        self.tabs = QTabWidget()
        splitter.addWidget(self.tabs)
        self._make_parameters_tab()
        self._make_theory_tab()
        self._make_data_tab()
        self._make_custom_tab()
        self._make_original_tab()
        parameter_font = self.parameters_edit.font()
        theory_size = self.theory_edit.font().pointSizeF()
        if theory_size > parameter_font.pointSizeF():
            parameter_font.setPointSizeF(theory_size)
            self.parameters_edit.setFont(parameter_font)

        right = QWidget()
        rv = QVBoxLayout(right)
        self.canvas = PlotCanvas()
        rv.addWidget(NavigationToolbar2QT(self.canvas, right))
        rv.addWidget(self.canvas, 1)
        rv.addWidget(QLabel("Log / fit results"))
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumHeight(145)
        rv.addWidget(self.console)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([720, 670])

    def _make_parameters_tab(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(2, 4, 2, 4)
        help_label = QLabel(
            "One parameter per line: number  name  value  step  min  max. "
            "Use 'none' for an unset boundary; step = 0 fixes a parameter. "
            "Blank lines and lines beginning with # are preserved.")
        help_label.setWordWrap(True)
        layout.addWidget(help_label)
        self.parameters_edit = QPlainTextEdit()
        self.parameters_edit.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.parameters_edit.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        layout.addWidget(self.parameters_edit, 1)
        buttons = QHBoxLayout()
        apply_button = QPushButton("Apply parameters")
        apply_button.clicked.connect(self.apply_parameters)
        buttons.addWidget(apply_button)
        renumber_button = QPushButton("Renumber sequentially")
        renumber_button.clicked.connect(self.renumber_parameters)
        buttons.addWidget(renumber_button)
        format_button = QPushButton("Align / format")
        format_button.clicked.connect(self.format_parameters)
        buttons.addWidget(format_button)
        buttons.addStretch(1)
        buttons.addWidget(QLabel("Significant digits:"))
        self.parameter_digits_spin = QSpinBox()
        self.parameter_digits_spin.setRange(1, 15)
        self.parameter_digits_spin.setValue(DEFAULT_SIGNIFICANT_DIGITS)
        buttons.addWidget(self.parameter_digits_spin)
        layout.addLayout(buttons)
        self.tabs.addTab(panel, "Parameters")

    def _make_theory_tab(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(QLabel("musrfit-style THEORY: consecutive lines multiply, '+' adds components."))
        picker = QHBoxLayout()
        picker.addWidget(QLabel("Add function:"))
        self.theory_function_combo = QComboBox()
        self.theory_function_combo.currentIndexChanged.connect(self._theory_function_selected)
        picker.addWidget(self.theory_function_combo, 1)
        insert = QPushButton("Insert into THEORY")
        insert.clicked.connect(self._insert_theory_function)
        picker.addWidget(insert)
        layout.addLayout(picker)
        self.theory_function_description = QLabel()
        self.theory_function_description.setWordWrap(True)
        layout.addWidget(self.theory_function_description)
        self.theory_edit = QPlainTextEdit()
        self.theory_edit.setTabStopDistance(24)
        layout.addWidget(self.theory_edit, 3)
        layout.addWidget(QLabel("Auxiliary FUNCTIONS (e.g. fun1 = gamma_mu * par5):"))
        self.functions_edit = QPlainTextEdit()
        layout.addWidget(self.functions_edit, 1)
        b = QPushButton("Apply theory and preview")
        b.clicked.connect(self.redraw)
        layout.addWidget(b)
        self.tabs.addTab(panel, "Theory")

    def _make_data_tab(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(QLabel("Dataset and RUN processing (original MSR settings are preserved)."))
        self.data_label = QPlainTextEdit()
        self.data_label.setReadOnly(True)
        layout.addWidget(self.data_label)
        b = QPushButton("Change data source…")
        b.clicked.connect(self.change_data_source_dialog)
        layout.addWidget(b)
        layout.addWidget(QLabel("Plot / fit window (μs):"))
        h = QHBoxLayout()
        self.tmin = QDoubleSpinBox()
        self.tmax = QDoubleSpinBox()
        for w in (self.tmin, self.tmax):
            w.setRange(-1000, 100000)
            w.setDecimals(4)
        h.addWidget(self.tmin)
        h.addWidget(self.tmax)
        layout.addLayout(h)
        self.tmin.editingFinished.connect(self.redraw)
        self.tmax.editingFinished.connect(self.redraw)

        corrections = QGroupBox("Run corrections")
        corrections_layout = QVBoxLayout(corrections)
        corrections_help = QLabel(
            "musrfit-compatible detector groups use one-based numbers or ranges (for example 1-3). "
            "For asymmetry, t0 values are ordered forward1 backward1 [forward2 backward2 ...].")
        corrections_help.setWordWrap(True)
        corrections_layout.addWidget(corrections_help)
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Fit data as:"))
        self.fit_type_combo = QComboBox()
        self.fit_type_combo.addItem("Forward/backward asymmetry", 2)
        self.fit_type_combo.addItem("Single histogram counts", 0)
        mode_row.addWidget(self.fit_type_combo, 1)
        corrections_layout.addLayout(mode_row)

        self.forward_edit = QLineEdit()
        self.backward_edit = QLineEdit()
        self.t0_edit = QLineEdit()
        self.background_edit = QLineEdit()
        self.data_range_edit = QLineEdit()
        for label, editor in (
            ("Forward detector(s)", self.forward_edit),
            ("Backward detector(s)", self.backward_edit),
            ("t0 bin(s)", self.t0_edit),
            ("Background bins", self.background_edit),
            ("Data bins", self.data_range_edit),
        ):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            row.addWidget(editor, 1)
            corrections_layout.addLayout(row)
        packing_row = QHBoxLayout()
        packing_row.addWidget(QLabel("Packing"))
        self.packing_spin = QSpinBox()
        self.packing_spin.setRange(1, 100000)
        packing_row.addWidget(self.packing_spin)
        packing_row.addStretch(1)
        corrections_layout.addLayout(packing_row)

        correction_buttons = QHBoxLayout()
        self.find_t0_button = QPushButton("Edit ranges visually…")
        self.apply_t0_button = QPushButton("Apply detector processing")
        self.reset_corrections_button = QPushButton("Reset corrections")
        self.find_t0_button.setEnabled(False)
        self.find_t0_button.setToolTip("Set t0, data and background ranges on the raw histograms")
        self.find_t0_button.clicked.connect(self.edit_corrections_visually)
        self.apply_t0_button.clicked.connect(self._apply_processing_controls)
        self.reset_corrections_button.clicked.connect(self._reset_processing_controls)
        correction_buttons.addWidget(self.find_t0_button)
        correction_buttons.addWidget(self.apply_t0_button)
        correction_buttons.addWidget(self.reset_corrections_button)
        corrections_layout.addLayout(correction_buttons)
        layout.addWidget(corrections)
        layout.addStretch(1)
        self.tabs.addTab(panel, "Runs / Data")

    def _make_custom_tab(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(QLabel("Write Python functions here; register with @register_function."))
        self.custom_edit = QPlainTextEdit()
        layout.addWidget(self.custom_edit)
        b = QPushButton("Activate code (only trusted Python!)")
        b.clicked.connect(self.activate_custom)
        layout.addWidget(b)
        self.registry_label = QLabel("")
        self.registry_label.setWordWrap(True)
        layout.addWidget(self.registry_label)
        layout.addWidget(QLabel("Python executes with your account permissions. Opening a project never auto-executes its code."))
        self.tabs.addTab(panel, "Custom models")

    def _make_original_tab(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(QLabel("Original imported .msr (read-only; unchanged in native project)."))
        self.original_edit = QPlainTextEdit()
        self.original_edit.setReadOnly(True)
        layout.addWidget(self.original_edit)
        self.tabs.addTab(panel, "Original .msr")

    def _log(self, text: str):
        self.console.appendPlainText(str(text))
        window = self.window()
        if isinstance(window, QMainWindow):
            window.statusBar().showMessage(str(text).splitlines()[0][:160])

    def _populate(self):
        significant_digits = int(self.project.display.get(
            "parameter_significant_digits", DEFAULT_SIGNIFICANT_DIGITS))
        self.parameter_digits_spin.setValue(significant_digits)
        parameter_text = self.project.parameter_text or format_parameter_text(
            self.project.parameters, significant_digits=significant_digits)
        self.project.parameter_text = parameter_text
        self.parameters_edit.setPlainText(parameter_text)
        self.theory_edit.setPlainText(self.project.theory)
        self.functions_edit.setPlainText(self.project.functions_text)
        self.custom_edit.setPlainText(self.project.custom_python or CUSTOM_EXAMPLE)
        self.original_edit.setPlainText(self.project.source_msr or "Not imported from an .msr file")
        self.tmin.setValue(float(self.project.display.get("t_start", 0)))
        self.tmax.setValue(float(self.project.display.get("t_stop", 8)))
        self._refresh_processing_controls()
        self._refresh_runs()
        self._refresh_registry()

    def _refresh_registry(self):
        self.registry_label.setText("Available models: " + ", ".join(sorted(x.name for x in REGISTRY.values())))
        selected = self.theory_function_combo.currentData()
        blocker = QSignalBlocker(self.theory_function_combo)
        self.theory_function_combo.clear()
        for key, spec in sorted(REGISTRY.items(), key=lambda item: item[1].name.lower()):
            self.theory_function_combo.addItem(spec.name, key)
        if selected is not None:
            index = self.theory_function_combo.findData(selected)
            if index >= 0:
                self.theory_function_combo.setCurrentIndex(index)
        del blocker
        self._theory_function_selected()

    def _theory_function_selected(self, *_):
        key = self.theory_function_combo.currentData()
        spec = REGISTRY.get(key) if key is not None else None
        if spec is None:
            self.theory_function_description.clear()
            return
        arguments = " ".join(f"par {n + 1}" for n in range(spec.arity)) or "no parameters"
        description = spec.description or "No description available"
        self.theory_function_description.setText(
            f"{description} · {spec.arity} argument(s): {arguments}")

    def _insert_theory_function(self):
        key = self.theory_function_combo.currentData()
        spec = REGISTRY.get(key) if key is not None else None
        if spec is None:
            return
        available = [parameter.index for parameter in self.project.parameters] or [1]
        arguments = [str(available[min(n, len(available) - 1)]) for n in range(spec.arity)]
        line = " ".join([spec.name, *arguments])
        cursor = self.theory_edit.textCursor()
        if not cursor.atBlockStart():
            cursor.insertText("\n")
        cursor.insertText(line)
        self.theory_edit.setTextCursor(cursor)
        self.theory_edit.setFocus()

    @staticmethod
    def _tokens(text):
        return text.replace(",", " ").split()

    def _refresh_processing_controls(self):
        run = self.project.runs[0] if self.project.runs else None
        if run is None:
            values = {"forward": [], "backward": [], "t0": [], "background": [], "data": []}
            fit_type, packing = 2, DEFAULT_PACKING
        else:
            values = {key: run.tokens(key) for key in ("forward", "backward", "t0", "background", "data")}
            fit_type = run.fittype
            packing = (
                int(float(run.tokens("packing")[0]))
                if run.tokens("packing") else DEFAULT_PACKING
            )
        index = self.fit_type_combo.findData(fit_type)
        self.fit_type_combo.setCurrentIndex(max(0, index))
        self.forward_edit.setText(" ".join(values["forward"]))
        self.backward_edit.setText(" ".join(values["backward"]))
        self.t0_edit.setText(" ".join(values["t0"]))
        self.background_edit.setText(" ".join(values["background"]))
        self.data_range_edit.setText(" ".join(values["data"]))
        self.packing_spin.setValue(max(1, packing))
        enabled = self.raw_data is not None
        self.find_t0_button.setEnabled(enabled)
        self.apply_t0_button.setEnabled(enabled)
        self.reset_corrections_button.setEnabled(enabled)

    def _default_run_for_raw_data(self, path: Path) -> RunConfig:
        return self.session.default_run_for_raw_data(path)

    def edit_corrections_visually(self):
        if self.raw_data is None:
            return
        try:
            if not self.project.runs:
                source = Path(self.raw_data.metadata.get("source", "raw-data"))
                self.project.runs.append(self._default_run_for_raw_data(source))
            run = self.project.runs[0]
            dialog = RunCorrectionsDialog(self.raw_data, run, self)
            if dialog.exec_() != dialog.Accepted:
                return
            dialog.apply_to(run)
            self._refresh_processing_controls()
            self._apply_processing_controls()
        except Exception as exc:
            self._error(f"Visual correction editor failed: {exc}")

    def _apply_processing_controls(self):
        if self.raw_data is None:
            return
        try:
            if not self.project.runs:
                source = Path(self.raw_data.metadata.get("source", "raw-data"))
                self.project.runs.append(self._default_run_for_raw_data(source))
            run = self.project.runs[0]
            run.fittype = int(self.fit_type_combo.currentData())
            entries = {
                "forward": self._tokens(self.forward_edit.text()),
                "backward": self._tokens(self.backward_edit.text()),
                "t0": self._tokens(self.t0_edit.text()),
                "background": self._tokens(self.background_edit.text()),
                "data": self._tokens(self.data_range_edit.text()),
                "packing": [str(self.packing_spin.value())],
            }
            if run.fittype == 0:
                entries["backward"] = []
                if len(entries["t0"]) == 2 * len(entries["forward"]):
                    entries["t0"] = entries["t0"][0::2]
                if len(entries["background"]) >= 4:
                    entries["background"] = entries["background"][:2]
                if len(entries["data"]) >= 4:
                    entries["data"] = entries["data"][:2]
            for key, value in entries.items():
                if value:
                    run.options[key] = value
                else:
                    run.options.pop(key, None)
            normalization = self.session.process_raw_data()
            if normalization is not None:
                self._populate_parameters_only()
                self._log(f"Added histogram normalization parameter {normalization} (N0)")
            self._refresh_processing_controls()
            self._refresh_runs()
            self.redraw()
        except Exception as exc:
            self._error(f"Detector processing failed: {exc}")

    def _reset_processing_controls(self):
        if self.raw_data is None:
            return
        source = Path(self.raw_data.metadata.get("source", "raw-data"))
        default = self._default_run_for_raw_data(source)
        if self.project.runs:
            self.project.runs[0] = default
        else:
            self.project.runs.append(default)
        self._refresh_processing_controls()
        self._apply_processing_controls()

    def _refresh_runs(self):
        lines = []
        for n, run in enumerate(self.project.runs, 1):
            lines.append(f"RUN {n}: {run.source} [{run.instrument}, {run.facility}, {run.format_hint}]")
            lines.append(f"  fittype={run.fittype}; forward={run.tokens('forward')}; backward={run.tokens('backward')}")
            lines.append(f"  t0={run.tokens('t0')}; background={run.tokens('background')}; data={run.tokens('data')}")
        if self.session.source_path is not None:
            lines.append(f"\nData source: {self.session.source_path}")
        if self.data is not None:
            lines.append(f"\nAttached fit-ready spectrum: {len(self.data.time_us)} points, "
                         f"time={self.data.time_us[0]:g}…{self.data.time_us[-1]:g} μs")
            lines.append(f"Reader: {self.data.metadata.get('reader', 'derived')}")
        elif self.raw_data is not None:
            lines.append(f"\nRaw histograms: {self.raw_data.counts.shape[0]} detectors × "
                         f"{self.raw_data.counts.shape[1]} channels")
            lines.append("Detectors: " + ", ".join(
                f"{detector.number}:{detector.name} (t0={detector.t0_bin})"
                for detector in self.raw_data.detectors))
            lines.append("Raw data loaded but detector processing has not produced fit-ready data")
        elif self.session.source_path is not None:
            state = "ready to load on demand" if self.session.source_path.is_file() else "file not found"
            lines.append(f"Data arrays not loaded ({state})")
        else:
            lines.append("\nNO MEASURED SPECTRUM ATTACHED — only preview, no fitting")
        self.data_label.setPlainText("\n".join(lines) or "No RUN block; import a 3-column spectrum")

    def apply_parameters(self):
        try:
            self._sync_parameters()
        except ValueError as exc:
            self._error(f"Invalid parameter text: {exc}")
            return
        self._log(f"Applied {len(self.project.parameters)} parameters")
        self.redraw()

    def renumber_parameters(self):
        try:
            renumbered = renumber_parameter_text(self.parameters_edit.toPlainText())
            parameters = parse_parameter_text(renumbered, self.project.parameters)
            text = format_parameter_text(
                parameters,
                renumbered,
                significant_digits=self.parameter_digits_spin.value(),
            )
        except ValueError as exc:
            self._error(f"Cannot renumber parameters: {exc}")
            return
        self.project.parameters = parameters
        self.project.parameter_text = text
        self.parameters_edit.setPlainText(text)
        self._log(
            "Parameters renumbered sequentially. Check THEORY/FUNCTIONS/RUN references if indices changed.")
        self.redraw()

    def format_parameters(self):
        try:
            self._sync_parameters()
            self._populate_parameters_only()
        except ValueError as exc:
            self._error(f"Cannot format parameters: {exc}")
            return
        self._log(
            f"Parameters aligned using {self.parameter_digits_spin.value()} significant digits")
        self.redraw()

    def _populate_parameters_only(self):
        template = self.parameters_edit.toPlainText()
        text = format_parameter_text(
            self.project.parameters,
            template,
            significant_digits=self.parameter_digits_spin.value(),
        )
        self.project.parameter_text = text
        self.parameters_edit.setPlainText(text)

    def _sync_parameters(self):
        text = self.parameters_edit.toPlainText()
        self.project.parameters = parse_parameter_text(text, self.project.parameters)
        self.project.parameter_text = text

    def _sync(self):
        self._sync_parameters()
        self.project.theory = self.theory_edit.toPlainText()
        self.project.functions_text = self.functions_edit.toPlainText()
        self.project.custom_python = self.custom_edit.toPlainText()
        self.project.display["t_start"] = float(self.tmin.value())
        self.project.display["t_stop"] = float(self.tmax.value())
        self.project.display["parameter_significant_digits"] = self.parameter_digits_spin.value()

    def open_path(self, path):
        try:
            path = Path(path).expanduser().resolve()
            if path.suffix.lower() == ".msr":
                project = read_msr(path)
            elif path.suffix.lower() == ".mysr":
                project = Project.load(path)
            else:
                raise ValueError("Expected a native .mysr project or musrfit .msr file")
            self.session.replace_project(project)
            self._populate()
            self._log(f"Opened {path.name}: {len(project.parameters)} parameters, {len(project.runs)} run(s)")
            self._try_auto_data()
            self.redraw()
            self.title_changed.emit()
        except Exception as exc:
            self._error(f"Cannot open {path}: {exc}")

    def _try_auto_data(self):
        candidate = self.session.discover_project_data_source()
        if candidate is None:
            if self.project.runs and self.project.runs[0].source:
                base = self.project.resolve(self.project.runs[0].source)
                self._log(f"Referenced raw file not found: {base} (original .msr imported successfully)")
            return
        if not candidate.is_file():
            self._log(f"Configured data source not found: {candidate}")
        else:
            self._log(f"Data source: {candidate.name} (loads on demand)")
        self._refresh_runs()

    def change_data_source_dialog(self):
        file, _ = QFileDialog.getOpenFileName(self, "Change μSR data source", "",
            "Data (*.txt *.csv *.dat *.nxs *.nexus *.h5 *.hdf5 *.bin *.mdu *.root *.mud);;All files (*)")
        if file:
            try:
                self._attach_data_source(Path(file))
                self._ensure_data_loaded()
                self.redraw()
            except Exception as exc:
                self._error(f"Data source failed: {exc}")

    def _attach_data_source(self, path, *, persist=True):
        source = self.session.attach_data_source(path, persist=persist)
        self._log(f"Data source: {source.name} (loads on demand)")
        self._refresh_processing_controls()
        self._refresh_runs()

    def _ensure_data_loaded(self, *, force=False):
        outcome = self.session.ensure_data_loaded(force=force)
        if not outcome.loaded:
            return False
        if outcome.reloaded and self.session.source_path is not None:
            self._log(f"Reloaded changed data source: {self.session.source_path.name}")
        if outcome.processing_error:
            self._log(f"Raw histograms loaded; configure detector processing: {outcome.processing_error}")
        if outcome.normalization_parameter is not None:
            self._populate_parameters_only()
            self._log(f"Added histogram normalization parameter {outcome.normalization_parameter} (N0)")
        self._refresh_processing_controls()
        self._refresh_runs()
        if self.data is not None and self.session.source_path is not None:
            self._log(
                f"Loaded {self.session.source_path.name}: {len(self.data.time_us)} fit-ready points")
        return True

    def save_dialog(self):
        try:
            self._sync()
        except Exception as exc:
            self._error(f"Cannot save: {exc}")
            return
        default = str(self.project.location.with_suffix(".mysr")) if self.project.location else "project.mysr"
        file, _ = QFileDialog.getSaveFileName(self, "Save native project", default, "mysr project (*.mysr)")
        if file:
            try:
                dest = Path(file).with_suffix(".mysr").resolve()
                # Adjust relative references so a saved .mysr keeps pointing to the source files.
                old_root = self.project.location.parent if self.project.location else Path.cwd()
                for key in ("data_override", "spectrum_path"):
                    val = getattr(self.project, key)
                    if val and not Path(val).is_absolute():
                        setattr(self.project, key, str((old_root / val).resolve()))
                for run in self.project.runs:
                    if run.source and not Path(run.source).is_absolute():
                        run.source = str((old_root / run.source).resolve())
                self.project.save(dest)
                self._log(f"Saved {dest}")
                self.title_changed.emit()
            except Exception as exc:
                self._error(f"Save failed: {exc}")

    def activate_custom(self):
        source = self.custom_edit.toPlainText()
        answer = QMessageBox.warning(self, "Run trusted Python code?",
            "This executes the custom Python source with full access to your computer.\n"
            "Only activate code you wrote or have reviewed.",
            QMessageBox.Ok | QMessageBox.Cancel, QMessageBox.Cancel)
        if answer != QMessageBox.Ok:
            return
        try:
            names = activate_user_code(source)
            self.project.custom_python = source
            self._log("Registered functions: " + (", ".join(names) or "none changed"))
            self._refresh_registry()
            self.redraw()
        except Exception as exc:
            self._error(f"Custom function activation failed: {exc}")

    def _mapping(self):
        return [int(float(x)) for x in self.project.runs[0].tokens("map")] if self.project.runs else []

    def redraw(self, *_):
        if not hasattr(self, "canvas"):
            return
        try:
            self._sync()
            self._ensure_data_loaded()
            if self.project.display["t_stop"] <= self.project.display["t_start"]:
                raise ValueError("Plot/fitting time max must exceed min")
            tmin = self.project.display["t_start"]
            tmax = self.project.display["t_stop"]
            t = np.linspace(tmin, tmax, 1400)
            if self.data is None:
                model = evaluate_theory(self.project.theory, t, self.project.parameter_values(),
                                        self.project.functions_text, self._mapping())
            else:
                model = evaluate_data_model(self.project, self.data, t)
            self.canvas.figure.clear()
            if self.mode.currentIndex() == 1:
                ax = self.canvas.figure.add_subplot(111)
                self._plot_fft(ax, t, model)
            elif self.data is not None and self.residual_check.isChecked():
                axes = self.canvas.figure.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]})
                ax, rx = axes
                self._plot_time(ax, t, model)
                data = self.data
                valid = (data.time_us >= tmin) & (data.time_us <= tmax)
                pts = data.time_us[valid]
                ymodel = evaluate_data_model(self.project, data, pts)
                rx.errorbar(pts, data.values[valid] - ymodel, yerr=data.error[valid],
                            fmt=".", markersize=2, linewidth=0.5)
                rx.axhline(0, color="grey", linewidth=0.8)
                rx.set_ylabel("Residual")
                rx.set_xlabel("Time (μs)")
            else:
                ax = self.canvas.figure.add_subplot(111)
                self._plot_time(ax, t, model)
                ax.set_xlabel("Time (μs)")
            self.canvas.draw_idle()
        except Exception as exc:
            self._log(f"Plot/model error: {exc}")

    def _plot_time(self, ax, t, model):
        if self.data is not None:
            sel = (self.data.time_us >= t[0]) & (self.data.time_us <= t[-1])
            # Thin rendered points for responsiveness; fit always uses full data.
            idx = np.flatnonzero(sel)
            idx = idx[::max(1, len(idx) // 1500)]
            ax.errorbar(self.data.time_us[idx], self.data.values[idx], yerr=self.data.error[idx],
                        fmt=".", color="0.35", ecolor="0.75", markersize=3,
                        linewidth=0.5, alpha=0.8, label="Data", zorder=2)
            title = self.project.title
        else:
            title = self.project.title + " — MODEL PREVIEW ONLY (NO DATA)"
        ax.plot(t, model, linewidth=1.8, label="Model", zorder=4)
        ylabel = "Counts" if isinstance(self.data, CountSpectrum) else "Asymmetry"
        ax.set(title=title, ylabel=ylabel)
        ax.grid(alpha=0.15)
        ax.legend(loc="best")

    def _plot_fft(self, ax, t, model):
        if self.data is not None:
            data = self.data
            valid = (data.time_us >= t[0]) & (data.time_us <= t[-1])
            tx = data.time_us[valid]
            yy = data.values[valid]
        else:
            tx, yy = t, model
        if len(tx) < 4:
            raise ValueError("Not enough data points for FFT")
        dt = np.median(np.diff(tx))
        if not np.allclose(np.diff(tx), dt, rtol=1e-3):
            raise ValueError("FFT requires uniformly spaced time samples")
        window = np.hanning(len(yy))
        frequencies = np.fft.rfftfreq(len(yy), dt)
        amplitude = abs(np.fft.rfft((yy - np.mean(yy)) * window))
        ax.plot(frequencies, amplitude)
        ax.set(xlabel="Frequency (MHz)", ylabel="FFT magnitude",
               title="Exploratory FFT (not musrfit-equivalent)")
        ax.grid(alpha=0.15)

    def fit(self, *_):
        try:
            self._sync()
            self._ensure_data_loaded()
        except Exception as exc:
            self._error(f"Cannot start fit: {exc}")
            return
        if self.data is None:
            self._error("Cannot fit: no data source is attached to this analysis.")
            return
        try:
            result = fit_project(self.project, self.data)
            if not result.valid:
                self._log("MINUIT did not converge to a valid minimum; parameters NOT applied")
                return
            for p in self.project.parameters:
                p.value = result.values[p.index]
                if not p.fixed:
                    p.step = result.errors[p.index]
            self._populate_parameters_only()
            self._log(f"MINUIT2: χ²={result.chi2:.5g}, NDF={result.ndf}, χ²/NDF={result.chi2/result.ndf:.5g}")
            self.redraw()
        except Exception as exc:
            self._error(f"Fit failed: {exc}")

    def export_plot(self):
        file, _ = QFileDialog.getSaveFileName(self, "Export figure", "mysr_plot.png", "PNG (*.png)")
        if file:
            self.canvas.figure.savefig(file, dpi=250)
            self._log(f"Saved plot: {file}")

    def _error(self, message):
        self._log(message)
        QMessageBox.warning(self, "mysr", message)
