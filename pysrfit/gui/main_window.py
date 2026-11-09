"""Single-window GUI: msr editor + model preview + data/fit tools."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import numpy as np
from PyQt5.QtCore import Qt, QSignalBlocker, QThread, pyqtSignal
from PyQt5.QtGui import QFontDatabase, QIntValidator
from PyQt5.QtWidgets import (QMainWindow, QWidget, QSplitter, QVBoxLayout, QHBoxLayout,
    QTabWidget, QPlainTextEdit, QLabel, QPushButton, QToolBar, QAction,
    QFileDialog, QMessageBox, QComboBox, QCheckBox, QGroupBox,
    QDoubleSpinBox, QFormLayout, QLineEdit, QSpinBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView, QProgressDialog,
    QScrollArea, QFrame, QSizePolicy)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure

from ..models import (
    DEFAULT_PACKING,
    FitRecord,
    Project,
    RunConfig,
    default_fourier_settings,
)
from ..msr import read_msr
from ..parameters import (
    DEFAULT_SIGNIFICANT_DIGITS,
    format_parameter_text,
    parse_parameter_text,
    renumber_parameter_text,
)
from ..run_numbers import (
    infer_run_number,
    neighboring_run_source,
    path_for_run_number,
)
from ..batch import sequential_existing_outputs, sequential_plan
from ..results import collect_project_results, write_results_table
from ..session import AnalysisSession
from ..core.fit import evaluate_data_model, fit_global
from ..core.fourier import OUTPUTS as FOURIER_OUTPUTS
from ..core.fourier import UNITS as FOURIER_UNITS
from ..core.fourier import WINDOWS as FOURIER_WINDOWS
from ..core.fourier import fourier_transform
from ..core.theory import evaluate_theory
from ..core.functions import REGISTRY, CUSTOM_EXAMPLE, activate_user_code
from ..io.base import CountSpectrum, HistogramRun, Spectrum
from .batch_dialogs import (
    CollectionOptionsDialog,
    SequentialFitDialog,
    SequentialFitWorker,
)
from .run_corrections import RunCorrectionsDialog


DATA_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "h")


class PlotCanvas(FigureCanvas):
    def __init__(self):
        self.figure = Figure(figsize=(8, 6), constrained_layout=True)
        super().__init__(self.figure)


class MainWindow(QMainWindow):
    def __init__(self, path: str | None = None):
        super().__init__()
        self.setWindowTitle("PySRFit — μSR analysis")
        self.resize(1390, 820)
        self._make_gui()
        self.add_analysis(path=path)

    def _make_gui(self):
        file_menu = self.menuBar().addMenu("&File")
        self._action(file_menu, "New analysis", self.add_analysis, "Ctrl+N")
        self._action(file_menu, "Open project / .msr in new tab...", self.open_dialog, "Ctrl+O")
        self._action(
            file_menu, "Add data files…",
            lambda: self._call("add_run_files_dialog"), "Ctrl+D")
        self._action(file_menu, "Collect fit results...", self.collect_results_dialog)
        file_menu.addSeparator()
        self._action(file_menu, "Save", lambda: self._call("save_project"), "Ctrl+S")
        self._action(
            file_menu, "Save As...", lambda: self._call("save_as_dialog"),
            "Ctrl+Shift+S")
        self._action(file_menu, "Export current plot as PNG...", lambda: self._call("export_plot"))
        self._action(file_menu, "Close analysis tab", self.close_current_analysis, "Ctrl+W")
        file_menu.addSeparator()
        self._action(file_menu, "Quit", self.close, "Ctrl+Q")

        analysis_menu = self.menuBar().addMenu("&Analysis")
        self._action(
            analysis_menu, "Change data source...",
            lambda: self._call("change_data_source_dialog"), "Ctrl+Shift+O")
        self._action(
            analysis_menu, "Apply parameters",
            lambda: self._call("apply_parameters"), "Ctrl+R")
        self._action(analysis_menu, "Fit", lambda: self._call("fit"), "Ctrl+F")

        toolbar = QToolBar("Analysis tabs")
        self.addToolBar(toolbar)
        for text, callback in [
            ("New analysis", self.add_analysis),
            ("Open project", self.open_dialog),
            ("Collect results", self.collect_results_dialog),
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

    def add_analysis(self, checked=False, *, path=None, data_path=None, data_paths=None):
        analysis = AnalysisWorkspace(path=path)
        analysis.title_changed.connect(lambda: self._refresh_tab_title(analysis))
        index = self.analysis_tabs.addTab(analysis, analysis.document_title())
        self.analysis_tabs.setCurrentIndex(index)
        sources = list(data_paths or ([] if data_path is None else [data_path]))
        if sources:
            try:
                for position, data_source in enumerate(sources):
                    source = Path(data_source).expanduser().resolve()
                    if position == 0:
                        analysis.project.title = source.stem
                    analysis._attach_data_source(source, new_run=position > 0)
                    analysis._ensure_data_loaded()
                analysis.redraw()
                analysis.title_changed.emit()
            except Exception as exc:
                analysis._error(f"Data import failed: {exc}")
        self._current_changed(index)
        return analysis

    def open_dialog(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Open PySRFit or musrfit file", "",
            "Projects and musrfit (*.pysrfit *.msr);;All files (*)")
        if file:
            self.add_analysis(path=file)

    def collect_results_dialog(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select fitted PySRFit projects",
            "",
            "PySRFit projects (*.pysrfit)",
        )
        if not files:
            return
        paths = [Path(path).expanduser().resolve() for path in files]
        dialog = CollectionOptionsDialog(paths, self)
        if dialog.exec_() != dialog.Accepted:
            return
        default = paths[0].parent / "collected_results.csv"
        output, _ = QFileDialog.getSaveFileName(
            self,
            "Save collected fit results",
            str(default),
            "Comma-separated values (*.csv);;All files (*)",
        )
        if not output:
            return
        try:
            schema, rows = collect_project_results(
                paths,
                dialog.axis,
                custom_field=dialog.custom_field,
            )
            destination = write_results_table(
                output,
                dialog.axis,
                schema,
                rows,
                custom_field=dialog.custom_field,
            )
        except Exception as exc:
            QMessageBox.warning(self, "Cannot collect fit results", str(exc))
            return
        self.statusBar().showMessage(
            f"Collected {len(rows)} fitted projects: {destination}")
        QMessageBox.information(
            self,
            "Fit results collected",
            f"Wrote {len(rows)} rows to:\n{destination}",
        )

    def close_current_analysis(self):
        index = self.analysis_tabs.currentIndex()
        if index >= 0:
            self.close_analysis(index)

    def close_analysis(self, index):
        widget = self.analysis_tabs.widget(index)
        if isinstance(widget, AnalysisWorkspace) and widget.batch_running:
            QMessageBox.warning(
                self,
                "Sequential fit running",
                "Cancel the sequential fit and wait for it to stop before closing this analysis.",
            )
            return
        self.analysis_tabs.removeTab(index)
        if widget is not None:
            widget.deleteLater()
        if self.analysis_tabs.count() == 0:
            self.add_analysis()

    def closeEvent(self, event):
        running = [
            self.analysis_tabs.widget(index)
            for index in range(self.analysis_tabs.count())
            if isinstance(self.analysis_tabs.widget(index), AnalysisWorkspace)
            and self.analysis_tabs.widget(index).batch_running
        ]
        if running:
            QMessageBox.warning(
                self,
                "Sequential fit running",
                "Cancel running sequential fits and wait for them to stop before quitting.",
            )
            event.ignore()
            return
        super().closeEvent(event)

    def _refresh_tab_title(self, analysis):
        index = self.analysis_tabs.indexOf(analysis)
        if index >= 0:
            self.analysis_tabs.setTabText(index, analysis.document_title())
        if analysis is self.current_analysis():
            self._current_changed(index)

    def _current_changed(self, _index):
        analysis = self.current_analysis()
        title = analysis.document_title() if analysis is not None else "μSR analysis"
        self.setWindowTitle(f"PySRFit — {title}")


class AnalysisWorkspace(QWidget):
    title_changed = pyqtSignal()

    def __init__(self, path: str | None = None):
        super().__init__()
        self.session = AnalysisSession()
        self._run_controls_busy = False
        self._metadata_table_busy = False
        self._fit_parameter_before_text: str | None = None
        self._fit_parameter_result_text: str | None = None
        self._fit_parameters_showing_result = False
        self._batch_thread = None
        self._batch_worker = None
        self._batch_progress = None
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

    @property
    def batch_running(self) -> bool:
        return self._batch_thread is not None

    def document_title(self):
        if self.project.location is not None:
            return self.project.location.name
        return self.project.title or "Untitled analysis"

    def _make_gui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        toolbar = QToolBar("Analysis controls")
        for text, fn in [
            ("Fit", self.fit),
            ("Sequential fit…", self.start_sequential_fit),
            ("Save", self.save_project),
        ]:
            b = QPushButton(text)
            b.clicked.connect(fn)
            toolbar.addWidget(b)
            if text.startswith("Sequential"):
                self.sequential_fit_button = b
        toolbar.addSeparator()
        self.mode = QComboBox()
        self.mode.addItems(["Time spectrum", "Fourier"])
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
        self._make_fourier_tab()
        self._make_custom_tab()
        self._make_original_tab()
        self._make_metadata_tab()
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
        view_packing_row = QHBoxLayout()
        view_packing_row.addWidget(QLabel("View packing:"))
        self.view_packing_spin = QSpinBox()
        self.view_packing_spin.setRange(1, 100000)
        self.view_packing_spin.setValue(DEFAULT_PACKING)
        self.view_packing_spin.setToolTip(
            "Packing used only for plotted raw-histogram data; fitting uses "
            "the packing configured in Runs / Data")
        self.view_packing_spin.valueChanged.connect(self.redraw)
        view_packing_row.addWidget(self.view_packing_spin)
        view_packing_row.addStretch(1)
        rv.addLayout(view_packing_row)
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
        self.revert_fit_button = QPushButton("Revert fit")
        self.revert_fit_button.setEnabled(False)
        self.revert_fit_button.clicked.connect(self.toggle_fit_parameters)
        buttons.addWidget(self.revert_fit_button)
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
        layout.addWidget(QLabel(
            "Add data files and configure the settings for each run."))
        run_row = QHBoxLayout()
        run_row.addWidget(QLabel("Selected run:"))
        self.run_combo = QComboBox()
        self.run_combo.currentIndexChanged.connect(self._run_selected)
        run_row.addWidget(self.run_combo, 1)
        self.run_plot_check = QCheckBox("Show in plot")
        self.run_plot_check.toggled.connect(self._run_plot_changed)
        run_row.addWidget(self.run_plot_check)
        self.run_details_check = QCheckBox("Show run details")
        run_row.addWidget(self.run_details_check)
        layout.addLayout(run_row)

        label_row = QHBoxLayout()
        label_row.addWidget(QLabel("Run label:"))
        self.run_label_edit = QLineEdit()
        self.run_label_edit.editingFinished.connect(self._run_label_changed)
        label_row.addWidget(self.run_label_edit, 1)
        layout.addLayout(label_row)

        run_number_row = QHBoxLayout()
        run_number_row.addWidget(QLabel("Run number:"))
        self.run_number_edit = QLineEdit()
        self.run_number_edit.setValidator(QIntValidator(0, 999999999, self))
        self.run_number_edit.setPlaceholderText("not detected")
        self.run_number_edit.setMaximumWidth(130)
        self.run_number_edit.returnPressed.connect(self.load_run_number)
        self.run_number_edit.textChanged.connect(self._run_number_text_changed)
        run_number_row.addWidget(self.run_number_edit)
        self.previous_run_button = QPushButton("Previous available")
        self.previous_run_button.clicked.connect(
            lambda: self.shift_run_number(-1))
        run_number_row.addWidget(self.previous_run_button)
        self.next_run_button = QPushButton("Next available")
        self.next_run_button.clicked.connect(lambda: self.shift_run_number(1))
        run_number_row.addWidget(self.next_run_button)
        self.load_run_button = QPushButton("Load run")
        self.load_run_button.clicked.connect(self.load_run_number)
        run_number_row.addWidget(self.load_run_button)
        run_number_row.addStretch(1)
        layout.addLayout(run_number_row)

        run_buttons = QHBoxLayout()
        add_runs = QPushButton("Add data files…")
        add_runs.clicked.connect(self.add_run_files_dialog)
        run_buttons.addWidget(add_runs)
        remove_run = QPushButton("Remove selected run")
        remove_run.clicked.connect(self.remove_selected_run)
        run_buttons.addWidget(remove_run)
        copy_settings = QPushButton("Copy corrections to all")
        copy_settings.clicked.connect(self.copy_selected_settings_to_all)
        run_buttons.addWidget(copy_settings)
        layout.addLayout(run_buttons)

        self.data_label = QPlainTextEdit()
        self.data_label.setReadOnly(True)
        self.data_label.setMaximumHeight(145)
        self.data_label.setVisible(False)
        self.run_details_check.toggled.connect(self.data_label.setVisible)
        layout.addWidget(self.data_label)

        plot_range = QGroupBox("Plot range")
        plot_range_layout = QHBoxLayout(plot_range)
        plot_range_layout.addWidget(QLabel("Time (μs):"))
        self.tmin = QDoubleSpinBox()
        self.tmax = QDoubleSpinBox()
        for w in (self.tmin, self.tmax):
            w.setRange(-1000, 100000)
            w.setDecimals(4)
            plot_range_layout.addWidget(w)
        self.tmin.editingFinished.connect(self.redraw)
        self.tmax.editingFinished.connect(self.redraw)

        plot_range_layout.addSpacing(8)
        plot_range_layout.addWidget(QLabel("Y:"))
        self.ymin = QDoubleSpinBox()
        self.ymax = QDoubleSpinBox()
        for widget in (self.ymin, self.ymax):
            widget.setRange(-1e12, 1e12)
            widget.setDecimals(6)
            widget.editingFinished.connect(self.redraw)
            plot_range_layout.addWidget(widget)
        self.y_auto_check = QCheckBox("Auto")
        self.y_auto_check.toggled.connect(self._plot_y_auto_changed)
        plot_range_layout.addWidget(self.y_auto_check)
        layout.addWidget(plot_range)

        corrections = QGroupBox("Run corrections")
        corrections_layout = QVBoxLayout(corrections)
        self.corrections_help = QLabel(
            "Forward/backward asymmetry fits use two detector groups and fit "
            "(alpha Nᶠ − Nᴮ) / (alpha Nᶠ + Nᴮ). Alpha references a fit-parameter "
            "number. Parameter maps, packing and fit ranges remain per run.")
        self.corrections_help.setWordWrap(True)
        self.corrections_help.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Minimum)
        corrections_layout.addWidget(self.corrections_help)
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Fit data as:"))
        self.fit_type_combo = QComboBox()
        self.fit_type_combo.addItem("Forward/backward asymmetry", 2)
        self.fit_type_combo.addItem("Single histogram counts", 0)
        self.fit_type_combo.currentIndexChanged.connect(self._fit_type_changed)
        mode_row.addWidget(self.fit_type_combo, 1)
        corrections_layout.addLayout(mode_row)

        self.forward_edit = QLineEdit()
        self.backward_edit = QLineEdit()
        self.t0_edit = QLineEdit()
        self.t0_edit.setPlaceholderText(
            "one bin applies to every selected detector")
        self.background_edit = QLineEdit()
        self.data_range_edit = QLineEdit()
        self.mapping_edit = QLineEdit()
        self.mapping_edit.setPlaceholderText("parameter numbers for map1 map2 …")
        self.alpha_edit = QLineEdit()
        self.alpha_edit.setPlaceholderText("fit parameter number")
        self.norm_edit = QLineEdit()
        self.norm_edit.setPlaceholderText("normalization fit parameter number")
        self.background_fit_edit = QLineEdit()
        self.background_fit_edit.setPlaceholderText("background fit parameter number")
        self.lifetime_edit = QLineEdit()
        self.lifetime_edit.setPlaceholderText("optional; blank uses 2.197019 μs")
        self.fit_range_edit = QLineEdit()
        self.fit_range_edit.setPlaceholderText("start stop; blank fits all prepared data")
        self.processing_rows = {}
        self.processing_labels = {}
        for key, label, editor in (
            ("forward", "Forward detector(s)", self.forward_edit),
            ("backward", "Backward detector(s)", self.backward_edit),
            ("background", "Background bins", self.background_edit),
            ("alpha", "Alpha parameter #", self.alpha_edit),
            ("norm", "Normalization parameter #", self.norm_edit),
            ("backgr.fit", "Background parameter #", self.background_fit_edit),
            ("lifetime", "Muon lifetime parameter #", self.lifetime_edit),
            ("t0", "t0 bin", self.t0_edit),
            ("data", "Data bins", self.data_range_edit),
            ("map", "Parameter map", self.mapping_edit),
            ("fit", "Fit range (μs)", self.fit_range_edit),
        ):
            row_widget = QWidget()
            row = QHBoxLayout(row_widget)
            row.setContentsMargins(0, 0, 0, 0)
            label_widget = QLabel(label)
            row.addWidget(label_widget)
            row.addWidget(editor, 1)
            corrections_layout.addWidget(row_widget)
            self.processing_rows[key] = row_widget
            self.processing_labels[key] = label_widget
        packing_row = QHBoxLayout()
        packing_row.addWidget(QLabel("Fit packing"))
        self.packing_spin = QSpinBox()
        self.packing_spin.setRange(1, 100000)
        self.packing_spin.setToolTip(
            "Packing used to prepare the data supplied to the fit")
        packing_row.addWidget(self.packing_spin)
        packing_row.addStretch(1)
        corrections_layout.addLayout(packing_row)

        correction_buttons = QHBoxLayout()
        self.find_t0_button = QPushButton("Edit ranges visually…")
        self.apply_t0_button = QPushButton("Apply selected run settings")
        self.reset_corrections_button = QPushButton("Reset corrections")
        self.find_t0_button.setEnabled(False)
        self.find_t0_button.setToolTip("Set t0, data and background ranges on the raw histograms")
        self.find_t0_button.clicked.connect(self.edit_corrections_visually)
        self.apply_t0_button.clicked.connect(self._apply_processing_controls)
        self.reset_corrections_button.clicked.connect(self._reset_processing_controls)
        for button in (
                self.find_t0_button,
                self.apply_t0_button,
                self.reset_corrections_button):
            button.setMinimumHeight(button.sizeHint().height())
            correction_buttons.addWidget(button)
        corrections_layout.addLayout(correction_buttons)
        layout.addWidget(corrections)
        layout.addStretch(1)

        # This tab can be taller than the available desktop on compact
        # displays.  Scrolling prevents Qt from squeezing the explanatory
        # text and bottom action buttons to only part of their normal height.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(panel)
        self.data_scroll_area = scroll
        self.tabs.addTab(scroll, "Runs / Data")
        self._update_fit_type_fields()

    def _make_metadata_tab(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        help_label = QLabel(
            "Metadata belongs to the selected run and is saved in the .pysrfit project. "
            "Values extracted from the data file are read-only; user values can be "
            "added and edited here. Histogram arrays remain in the external data file.")
        help_label.setWordWrap(True)
        layout.addWidget(help_label)

        self.metadata_table = QTableWidget(0, 3)
        self.metadata_table.setHorizontalHeaderLabels(["Field", "Value", "Source"])
        self.metadata_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.metadata_table.setSelectionMode(QAbstractItemView.SingleSelection)
        header = self.metadata_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.metadata_table.itemChanged.connect(self._metadata_item_changed)
        layout.addWidget(self.metadata_table, 1)

        buttons = QHBoxLayout()
        add_button = QPushButton("Add user field")
        add_button.clicked.connect(self._add_user_metadata)
        buttons.addWidget(add_button)
        remove_button = QPushButton("Remove selected user field")
        remove_button.clicked.connect(self._remove_user_metadata)
        buttons.addWidget(remove_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self.tabs.addTab(panel, "Metadata")

    def _make_fourier_tab(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        help_label = QLabel(
            "Fourier transform of every run marked Show in plot. "
            "The displayed time range supplies the transform input.")
        help_label.setWordWrap(True)
        layout.addWidget(help_label)

        form = QFormLayout()
        self.fourier_window_combo = QComboBox()
        self.fourier_window_combo.addItems(FOURIER_WINDOWS)
        form.addRow("Window", self.fourier_window_combo)
        self.fourier_output_combo = QComboBox()
        self.fourier_output_combo.addItems(FOURIER_OUTPUTS)
        form.addRow("Displayed quantity", self.fourier_output_combo)
        self.fourier_units_combo = QComboBox()
        self.fourier_units_combo.addItems(FOURIER_UNITS)
        form.addRow("Frequency units", self.fourier_units_combo)

        self.fourier_padding_spin = QSpinBox()
        self.fourier_padding_spin.setRange(1, 64)
        form.addRow("Zero-padding factor", self.fourier_padding_spin)
        self.fourier_phase_spin = QDoubleSpinBox()
        self.fourier_phase_spin.setRange(-3600, 3600)
        self.fourier_phase_spin.setDecimals(3)
        self.fourier_phase_spin.setSuffix("°")
        form.addRow("Phase rotation", self.fourier_phase_spin)

        frequency_range = QWidget()
        frequency_layout = QHBoxLayout(frequency_range)
        frequency_layout.setContentsMargins(0, 0, 0, 0)
        self.fourier_min_spin = QDoubleSpinBox()
        self.fourier_max_spin = QDoubleSpinBox()
        for widget in (self.fourier_min_spin, self.fourier_max_spin):
            widget.setRange(-1e9, 1e9)
            widget.setDecimals(5)
            frequency_layout.addWidget(widget)
        form.addRow("Displayed min/max", frequency_range)
        layout.addLayout(form)

        self.fourier_subtract_mean_check = QCheckBox("Subtract mean before transforming")
        layout.addWidget(self.fourier_subtract_mean_check)
        self.fourier_normalize_check = QCheckBox("Normalize transform amplitude")
        layout.addWidget(self.fourier_normalize_check)

        apply_button = QPushButton("Apply settings and show Fourier plot")
        apply_button.clicked.connect(self.apply_fourier_settings)
        layout.addWidget(apply_button)
        layout.addStretch(1)
        self.tabs.addTab(panel, "Fourier")

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

    def apply_fourier_settings(self):
        try:
            self._sync_fourier_settings()
            if self.project.fourier["frequency_max"] <= self.project.fourier["frequency_min"]:
                raise ValueError("Fourier frequency max must exceed min")
        except ValueError as exc:
            self._error(str(exc))
            return
        self.mode.setCurrentIndex(1)
        self.redraw()

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
        self.ymin.setValue(float(self.project.display.get("y_min", -0.3)))
        self.ymax.setValue(float(self.project.display.get("y_max", 0.3)))
        fit_packing = DEFAULT_PACKING
        if self.project.runs and self.project.runs[0].tokens("packing"):
            try:
                fit_packing = int(float(self.project.runs[0].tokens("packing")[0]))
            except ValueError:
                pass
        blocker = QSignalBlocker(self.view_packing_spin)
        self.view_packing_spin.setValue(max(
            1, int(self.project.display.get("view_packing", fit_packing))))
        del blocker
        blocker = QSignalBlocker(self.y_auto_check)
        self.y_auto_check.setChecked(bool(self.project.display.get("y_auto", True)))
        del blocker
        self._set_y_range_controls()
        fourier = default_fourier_settings()
        fourier.update(self.project.fourier)
        self.project.fourier = fourier
        self.fourier_window_combo.setCurrentText(str(fourier["window"]))
        self.fourier_output_combo.setCurrentText(str(fourier["output"]))
        self.fourier_units_combo.setCurrentText(str(fourier["units"]))
        self.fourier_padding_spin.setValue(int(fourier["zero_padding"]))
        self.fourier_phase_spin.setValue(float(fourier["phase_deg"]))
        self.fourier_min_spin.setValue(float(fourier["frequency_min"]))
        self.fourier_max_spin.setValue(float(fourier["frequency_max"]))
        self.fourier_subtract_mean_check.setChecked(bool(fourier["subtract_mean"]))
        self.fourier_normalize_check.setChecked(bool(fourier["normalize"]))
        self._refresh_run_selector()
        self._refresh_processing_controls()
        self._refresh_runs()
        self._refresh_metadata_table()
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

    def _refresh_run_selector(self):
        self._run_controls_busy = True
        try:
            blocker = QSignalBlocker(self.run_combo)
            self.run_combo.clear()
            for index, run in enumerate(self.project.runs):
                source_name = Path(run.source).name if run.source else "no source"
                label = run.label or source_name
                plot_state = "" if run.plot_enabled else " [not plotted]"
                self.run_combo.addItem(
                    f"{index + 1}: {label} — {source_name}{plot_state}")
            if self.project.runs:
                self.run_combo.setCurrentIndex(self.session.selected_run_index)
            del blocker
        finally:
            self._run_controls_busy = False

    def _run_selected(self, index):
        if self._run_controls_busy or index < 0 or index >= len(self.project.runs):
            return
        self.session.select_run(index)
        self._refresh_processing_controls()
        self._refresh_runs()
        self._refresh_metadata_table()

    def _selected_source_path(self) -> Path | None:
        run = self.session.current_run
        if run is None:
            return None
        state_path = self.session.source_path
        if state_path is not None:
            return state_path
        reference = run.source
        if self.session.selected_run_index == 0 and self.project.data_override:
            reference = self.project.data_override
        return self.project.resolve(reference) if reference else None

    def _refresh_run_number_controls(self):
        run = self.session.current_run
        source = self._selected_source_path()
        number = (
            infer_run_number(source, run.file_metadata)
            if run is not None and source is not None else None
        )
        blocker = QSignalBlocker(self.run_number_edit)
        self.run_number_edit.setText("" if number is None else str(number))
        del blocker
        self.run_number_edit.setEnabled(source is not None)
        self._run_number_text_changed(self.run_number_edit.text())
        try:
            previous = (
                neighboring_run_source(source, -1, run.file_metadata)
                if run is not None and source is not None else None
            )
            following = (
                neighboring_run_source(source, 1, run.file_metadata)
                if run is not None and source is not None else None
            )
        except OSError:
            previous = following = None
        self.previous_run_button.setEnabled(previous is not None)
        self.next_run_button.setEnabled(following is not None)
        self.previous_run_button.setToolTip(
            "No earlier matching data file"
            if previous is None else
            f"Load run {previous[0]}: {previous[1].name}")
        self.next_run_button.setToolTip(
            "No later matching data file"
            if following is None else
            f"Load run {following[0]}: {following[1].name}")

    def _run_number_text_changed(self, text):
        valid = (
            self._selected_source_path() is not None
            and text.isdigit() and int(text) >= 0
        )
        self.load_run_button.setEnabled(valid)

    def shift_run_number(self, direction):
        run = self.session.current_run
        source = self._selected_source_path()
        if run is None or source is None:
            return
        try:
            neighbor = neighboring_run_source(
                source, int(direction), run.file_metadata)
            if neighbor is None:
                label = "earlier" if direction < 0 else "later"
                self._log(f"No {label} matching run file is available")
                self._refresh_run_number_controls()
                return
            number, target = neighbor
            self._load_run_source(target, number)
        except Exception as exc:
            self._refresh_run_number_controls()
            self._error(f"Cannot navigate to another run: {exc}")

    def load_run_number(self):
        run = self.session.current_run
        source = self._selected_source_path()
        text = self.run_number_edit.text().strip()
        if run is None or source is None:
            return
        if not text.isdigit():
            self._error("Run number must be a non-negative integer")
            return
        try:
            target = path_for_run_number(source, int(text), run.file_metadata)
            self._load_run_source(target, int(text))
        except Exception as exc:
            self._refresh_run_number_controls()
            self._error(f"Cannot load run {text}: {exc}")

    def _load_run_source(self, target: Path, run_number: int):
        run = self.session.current_run
        source = self._selected_source_path()
        if run is None or source is None:
            return
        if not target.is_file():
            raise FileNotFoundError(f"Run data file not found: {target}")
        if target.resolve() == source.resolve():
            self._refresh_run_number_controls()
            self._log(f"Run {run_number} is already loaded")
            return
        update_default_label = not run.label or run.label == source.stem
        self._attach_data_source(target)
        if update_default_label:
            self.session.current_run.label = target.stem
        self._ensure_data_loaded(force=True)
        self._refresh_run_selector()
        self._refresh_processing_controls()
        self._refresh_runs()
        self._refresh_metadata_table()
        self.redraw()
        self._log(
            f"Changed selected data to run {run_number}: {target.name}")

    @staticmethod
    def _flatten_metadata(metadata, prefix=""):
        rows = []
        for key in sorted(metadata, key=lambda item: str(item).casefold()):
            value = metadata[key]
            name = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict) and value:
                rows.extend(AnalysisWorkspace._flatten_metadata(value, name))
            elif isinstance(value, (list, tuple, dict)):
                rows.append((name, json.dumps(value, ensure_ascii=False)))
            elif value is not None:
                rows.append((name, str(value)))
        return rows

    def _refresh_metadata_table(self):
        if not hasattr(self, "metadata_table"):
            return
        self._metadata_table_busy = True
        try:
            self.metadata_table.setRowCount(0)
            run = self.session.current_run
            if run is None:
                return
            file_rows = []
            for key, value in (
                ("facility", run.facility),
                ("instrument", run.instrument),
                ("format", run.format_hint),
            ):
                if value:
                    file_rows.append((key, str(value)))
            file_rows.extend(self._flatten_metadata(run.file_metadata))
            rows = [(*row, "Data file") for row in file_rows]
            rows.extend(
                (key, value, "User")
                for key, value in sorted(
                    run.user_metadata.items(), key=lambda item: item[0].casefold())
            )
            self.metadata_table.setRowCount(len(rows))
            read_only = Qt.ItemIsSelectable | Qt.ItemIsEnabled
            for row, (key, value, source) in enumerate(rows):
                key_item = QTableWidgetItem(key)
                value_item = QTableWidgetItem(str(value))
                source_item = QTableWidgetItem(source)
                source_item.setFlags(read_only)
                if source == "User":
                    key_item.setData(Qt.UserRole, key)
                else:
                    key_item.setFlags(read_only)
                    value_item.setFlags(read_only)
                self.metadata_table.setItem(row, 0, key_item)
                self.metadata_table.setItem(row, 1, value_item)
                self.metadata_table.setItem(row, 2, source_item)
        finally:
            self._metadata_table_busy = False

    def _metadata_item_changed(self, item):
        if self._metadata_table_busy or item.column() not in (0, 1):
            return
        run = self.session.current_run
        source = self.metadata_table.item(item.row(), 2)
        key_item = self.metadata_table.item(item.row(), 0)
        value_item = self.metadata_table.item(item.row(), 1)
        if run is None or source is None or source.text() != "User":
            return
        old_key = str(key_item.data(Qt.UserRole) or key_item.text()).strip()
        new_key = key_item.text().strip()
        if not new_key:
            self._error("Metadata field names cannot be blank")
            self._refresh_metadata_table()
            return
        file_keys = {key for key, _value in self._flatten_metadata(run.file_metadata)}
        file_keys.update({"facility", "instrument", "format"})
        if (new_key != old_key
                and (new_key in run.user_metadata or new_key in file_keys)):
            self._error(f"Metadata field {new_key!r} already exists")
            self._refresh_metadata_table()
            return
        value = value_item.text()
        if new_key != old_key:
            run.user_metadata.pop(old_key, None)
        run.user_metadata[new_key] = value
        self._refresh_metadata_table()

    def _add_user_metadata(self):
        run = self.session.current_run
        if run is None:
            return
        existing = set(run.user_metadata)
        existing.update(key for key, _value in self._flatten_metadata(run.file_metadata))
        key = "note"
        suffix = 2
        while key in existing:
            key = f"note_{suffix}"
            suffix += 1
        run.user_metadata[key] = ""
        self._refresh_metadata_table()
        for row in range(self.metadata_table.rowCount()):
            if (self.metadata_table.item(row, 0).text() == key
                    and self.metadata_table.item(row, 2).text() == "User"):
                self.metadata_table.setCurrentCell(row, 0)
                self.metadata_table.editItem(self.metadata_table.item(row, 0))
                break

    def _remove_user_metadata(self):
        run = self.session.current_run
        row = self.metadata_table.currentRow()
        if run is None or row < 0:
            return
        source = self.metadata_table.item(row, 2)
        key_item = self.metadata_table.item(row, 0)
        if source is None or source.text() != "User":
            self._log("File metadata is read-only")
            return
        run.user_metadata.pop(str(key_item.data(Qt.UserRole) or key_item.text()), None)
        self._refresh_metadata_table()

    def _run_plot_changed(self, plotted):
        if self._run_controls_busy or self.session.current_run is None:
            return
        self.session.current_run.plot_enabled = bool(plotted)
        self._refresh_run_selector()
        self.redraw()

    def _set_y_range_controls(self):
        manual = not self.y_auto_check.isChecked()
        self.ymin.setEnabled(manual)
        self.ymax.setEnabled(manual)

    def _plot_y_auto_changed(self, _checked):
        self._set_y_range_controls()
        self.redraw()

    def _run_label_changed(self):
        run = self.session.current_run
        if run is None:
            return
        run.label = self.run_label_edit.text().strip()
        self._refresh_run_selector()
        self.title_changed.emit()

    def add_run_files_dialog(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Add μSR data files to the current analysis",
            "",
            "Data (*.txt *.csv *.dat *.nxs *.nexus *.h5 *.hdf5 *.bin *.mdu *.root *.mud *.msr);;All files (*)",
        )
        if files:
            self.add_run_files(files)

    def add_run_files(self, paths):
        added = 0
        for path in paths:
            try:
                source = Path(path).expanduser().resolve()
                self._attach_data_source(source, new_run=bool(self.project.runs))
                self._ensure_data_loaded()
                added += 1
            except Exception as exc:
                self._log(f"Cannot add run {path}: {exc}")
        if added:
            self._refresh_run_selector()
            self._refresh_processing_controls()
            self._refresh_runs()
            self.redraw()
            self._log(f"Added {added} run(s); global analysis now has {len(self.project.runs)} run(s)")

    def remove_selected_run(self):
        if not self.project.runs:
            return
        removed = self.session.current_run
        self.session.remove_run()
        self._refresh_run_selector()
        self._refresh_processing_controls()
        self._refresh_runs()
        self._refresh_metadata_table()
        self.redraw()
        self._log(f"Removed run: {removed.label or removed.source}")

    def copy_selected_settings_to_all(self):
        source = self.session.current_run
        if source is None or len(self.project.runs) < 2:
            return
        processing_keys = {
            "forward", "backward", "t0", "background", "backgr.fix",
            "backgr.fit", "data", "packing", "alpha", "norm", "lifetime",
            "fit",
        }
        for index, run in enumerate(self.project.runs):
            if run is source:
                continue
            run.fittype = source.fittype
            for key in processing_keys:
                if source.tokens(key):
                    run.options[key] = list(source.tokens(key))
                else:
                    run.options.pop(key, None)
            if self.session.state(index).raw_data is not None:
                try:
                    self.session.process_raw_data(index=index)
                except Exception as exc:
                    self._log(f"Run {index + 1} could not be reprocessed: {exc}")
        self._refresh_runs()
        self.redraw()
        self._log("Copied detector corrections and fit settings to all runs")

    def _fit_type_changed(self, *_):
        self._update_fit_type_fields()

    def _update_fit_type_fields(self):
        if not hasattr(self, "processing_rows"):
            return
        single_histogram = self.fit_type_combo.currentData() == 0
        for key in ("backward", "background", "alpha"):
            self.processing_rows[key].setVisible(not single_histogram)
        for key in ("norm", "backgr.fit", "lifetime"):
            self.processing_rows[key].setVisible(single_histogram)
        self.processing_labels["forward"].setText(
            "Histogram detector(s)" if single_histogram else "Forward detector(s)")
        self.find_t0_button.setToolTip(
            "Set t0 and data range on the raw histogram"
            if single_histogram else
            "Set t0, data and background ranges on the raw histograms")
        if single_histogram:
            self.corrections_help.setText(
                "Single-histogram fits use one detector group and fit raw counts as "
                "N0 exp(-t/tau) [1 + A(t)] + background. Normalization and background "
                "reference fit-parameter numbers; lifetime is optional. Parameter maps, "
                "packing and fit ranges remain per run.")
        else:
            self.corrections_help.setText(
                "Forward/backward asymmetry fits use two detector groups and fit "
                "(alpha Nᶠ − Nᴮ) / (alpha Nᶠ + Nᴮ). Alpha references a "
                "fit-parameter number. Parameter maps, packing and fit ranges remain "
                "per run.")

    def _refresh_processing_controls(self):
        run = self.session.current_run
        if run is None:
            values = {
                "forward": [], "backward": [], "t0": [], "background": [],
                "data": [], "alpha": [], "norm": [], "backgr.fit": [],
                "lifetime": [], "map": [], "fit": [],
            }
            fit_type, packing = 2, DEFAULT_PACKING
        else:
            values = {
                key: run.tokens(key)
                for key in (
                    "forward", "backward", "t0", "background", "data",
                    "alpha", "norm", "backgr.fit", "lifetime", "map", "fit",
                )
            }
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
        self.alpha_edit.setText(" ".join(values["alpha"]))
        self.norm_edit.setText(" ".join(values["norm"]))
        self.background_fit_edit.setText(" ".join(values["backgr.fit"]))
        self.lifetime_edit.setText(" ".join(values["lifetime"]))
        self.mapping_edit.setText(" ".join(values["map"]))
        self.fit_range_edit.setText(" ".join(values["fit"]))
        self.packing_spin.setValue(max(1, packing))
        self._update_fit_type_fields()
        self._run_controls_busy = True
        try:
            self.run_label_edit.setText(run.label if run is not None else "")
            blocker = QSignalBlocker(self.run_plot_check)
            self.run_plot_check.setChecked(run.plot_enabled if run is not None else False)
            del blocker
        finally:
            self._run_controls_busy = False
        has_raw_data = self.raw_data is not None
        self.find_t0_button.setEnabled(has_raw_data)
        self.apply_t0_button.setEnabled(run is not None)
        self.reset_corrections_button.setEnabled(has_raw_data)
        self._refresh_run_number_controls()

    def _default_run_for_raw_data(self, path: Path) -> RunConfig:
        return self.session.default_run_for_raw_data(path)

    def edit_corrections_visually(self):
        if self.raw_data is None:
            return
        try:
            run = self.session.current_run
            if run is None:
                return
            dialog = RunCorrectionsDialog(self.raw_data, run, self)
            if dialog.exec_() != dialog.Accepted:
                return
            dialog.apply_to(run)
            self._refresh_processing_controls()
            self._apply_processing_controls()
        except Exception as exc:
            self._error(f"Visual correction editor failed: {exc}")

    def _apply_processing_controls(self):
        try:
            run = self.session.current_run
            if run is None:
                return
            run.label = self.run_label_edit.text().strip()
            previous_fit_type = run.fittype
            run.fittype = int(self.fit_type_combo.currentData())
            entries = {
                "forward": self._tokens(self.forward_edit.text()),
                "backward": self._tokens(self.backward_edit.text()),
                "t0": self._tokens(self.t0_edit.text()),
                "background": self._tokens(self.background_edit.text()),
                "data": self._tokens(self.data_range_edit.text()),
                "alpha": self._tokens(self.alpha_edit.text()),
                "norm": self._tokens(self.norm_edit.text()),
                "backgr.fit": self._tokens(self.background_fit_edit.text()),
                "lifetime": self._tokens(self.lifetime_edit.text()),
                "map": self._tokens(self.mapping_edit.text()),
                "fit": self._tokens(self.fit_range_edit.text()),
                "packing": [str(self.packing_spin.value())],
            }
            if run.fittype == 0:
                entries["backward"] = []
                entries["alpha"] = []
                entries["background"] = []
                run.options.pop("backgr.fix", None)
                if previous_fit_type == 2 and len(entries["t0"]) % 2 == 0:
                    entries["t0"] = entries["t0"][0::2]
                if len(entries["data"]) >= 4:
                    entries["data"] = entries["data"][:2]
            else:
                entries["norm"] = []
                entries["backgr.fit"] = []
                entries["lifetime"] = []
            for key, value in entries.items():
                if value:
                    run.options[key] = value
                else:
                    run.options.pop(key, None)
            added_parameters = (
                self.session.process_raw_data() if self.raw_data is not None else ())
            self._report_added_parameters(added_parameters)
            self._refresh_processing_controls()
            self._refresh_run_selector()
            self._refresh_runs()
            self.redraw()
        except Exception as exc:
            self._error(f"Detector processing failed: {exc}")

    def _reset_processing_controls(self):
        if self.raw_data is None:
            return
        source = Path(self.raw_data.metadata.get("source", "raw-data"))
        default = self._default_run_for_raw_data(source)
        self.project.runs[self.session.selected_run_index] = default
        self._refresh_processing_controls()
        self._apply_processing_controls()

    def _refresh_runs(self):
        lines = []
        for n, run in enumerate(self.project.runs, 1):
            selected = " ← selected" if n - 1 == self.session.selected_run_index else ""
            plot_state = "shown" if run.plot_enabled else "not shown"
            lines.append(
                f"RUN {n}: {run.label or Path(run.source).name} ({plot_state}){selected}\n"
                f"  {run.source} [{run.instrument}, {run.facility}, {run.format_hint}]")
            lines.append(f"  fittype={run.fittype}; forward={run.tokens('forward')}; backward={run.tokens('backward')}")
            lines.append(f"  t0={run.tokens('t0')}; background={run.tokens('background')}; data={run.tokens('data')}")
            if run.fittype == 0:
                lines.append(
                    f"  norm={run.tokens('norm')}; backgr.fit={run.tokens('backgr.fit')}; "
                    f"lifetime={run.tokens('lifetime') or ['default 2.197019 μs']}")
            else:
                alpha = run.tokens("alpha")
                alpha_description = str(alpha)
                if alpha:
                    try:
                        parameter = self.session.alpha_parameter(n - 1)
                        if parameter is not None:
                            alpha_description = (
                                f"parameter {parameter.index} ({parameter.name}) = {parameter.value:g}")
                    except ValueError as exc:
                        alpha_description = f"invalid ({exc})"
                lines.append(f"  alpha={alpha_description}")
            lines.append(
                f"  map={run.tokens('map')}; fit={run.tokens('fit')}; "
                f"packing={run.tokens('packing')}")
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
            refreshed = self.session.refresh_parameter_dependent_data()
        except ValueError as exc:
            self._error(f"Cannot apply parameters: {exc}")
            return
        suffix = f"; refreshed alpha for {refreshed} run(s)" if refreshed else ""
        self._log(f"Applied {len(self.project.parameters)} parameters{suffix}")
        self._refresh_runs()
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

    def _remember_fit_parameters(self, before_text: str, result_text: str):
        self._fit_parameter_before_text = before_text
        self._fit_parameter_result_text = result_text
        self._fit_parameters_showing_result = True
        self.revert_fit_button.setText("Revert fit")
        self.revert_fit_button.setEnabled(True)

    def toggle_fit_parameters(self):
        if (self._fit_parameter_before_text is None
                or self._fit_parameter_result_text is None):
            return
        restoring_result = not self._fit_parameters_showing_result
        text = (
            self._fit_parameter_result_text
            if restoring_result else self._fit_parameter_before_text
        )
        try:
            parameters = parse_parameter_text(text, self.project.parameters)
            self.project.parameters = parameters
            self.project.parameter_text = text
            self.parameters_edit.setPlainText(text)
            refreshed = self.session.refresh_parameter_dependent_data()
        except ValueError as exc:
            self._error(f"Cannot restore fit parameters: {exc}")
            return
        self._fit_parameters_showing_result = restoring_result
        self.revert_fit_button.setText(
            "Revert fit" if restoring_result else "Restore fit")
        label = "fit result" if restoring_result else "pre-fit parameters"
        suffix = f"; refreshed alpha for {refreshed} run(s)" if refreshed else ""
        self._log(f"Restored {label}{suffix}")
        self._refresh_runs()
        self.redraw()

    def _sync(self):
        self._sync_parameters()
        self._sync_fourier_settings()
        self.project.theory = self.theory_edit.toPlainText()
        self.project.functions_text = self.functions_edit.toPlainText()
        self.project.custom_python = self.custom_edit.toPlainText()
        self.project.display["t_start"] = float(self.tmin.value())
        self.project.display["t_stop"] = float(self.tmax.value())
        self.project.display["y_auto"] = self.y_auto_check.isChecked()
        self.project.display["y_min"] = float(self.ymin.value())
        self.project.display["y_max"] = float(self.ymax.value())
        self.project.display["view_packing"] = self.view_packing_spin.value()
        self.project.display["parameter_significant_digits"] = self.parameter_digits_spin.value()

    def _sync_fourier_settings(self):
        self.project.fourier = {
            "window": self.fourier_window_combo.currentText(),
            "subtract_mean": self.fourier_subtract_mean_check.isChecked(),
            "zero_padding": self.fourier_padding_spin.value(),
            "phase_deg": float(self.fourier_phase_spin.value()),
            "output": self.fourier_output_combo.currentText(),
            "normalize": self.fourier_normalize_check.isChecked(),
            "units": self.fourier_units_combo.currentText(),
            "frequency_min": float(self.fourier_min_spin.value()),
            "frequency_max": float(self.fourier_max_spin.value()),
        }

    def open_path(self, path):
        try:
            path = Path(path).expanduser().resolve()
            if path.suffix.lower() == ".msr":
                project = read_msr(path)
            elif path.suffix.lower() == ".pysrfit":
                project = Project.load(path)
            else:
                raise ValueError("Expected a native .pysrfit project or musrfit .msr file")
            self.session.replace_project(project)
            self._populate()
            self._log(f"Opened {path.name}: {len(project.parameters)} parameters, {len(project.runs)} run(s)")
            self._try_auto_data()
            self.redraw()
            self.title_changed.emit()
        except Exception as exc:
            self._error(f"Cannot open {path}: {exc}")

    def _try_auto_data(self):
        candidates = self.session.discover_project_data_sources()
        for index, candidate in enumerate(candidates):
            if candidate is None:
                continue
            if not candidate.is_file():
                self._log(f"Run {index + 1} data source not found: {candidate}")
            else:
                self._log(f"Run {index + 1} data source: {candidate.name} (loads on demand)")
        self._refresh_run_selector()
        self._refresh_processing_controls()
        self._refresh_runs()

    def change_data_source_dialog(self):
        file, _ = QFileDialog.getOpenFileName(self, "Change μSR data source", "",
            "Data (*.txt *.csv *.dat *.nxs *.nexus *.h5 *.hdf5 *.bin *.mdu *.root *.mud *.msr);;All files (*)")
        if file:
            try:
                self._attach_data_source(Path(file))
                self._ensure_data_loaded()
                self.redraw()
            except Exception as exc:
                self._error(f"Data source failed: {exc}")

    def _attach_data_source(self, path, *, persist=True, new_run=False):
        source = self.session.attach_data_source(
            path, persist=persist, new_run=new_run)
        self._log(f"Data source: {source.name} (loads on demand)")
        self._refresh_run_selector()
        self._refresh_processing_controls()
        self._refresh_runs()
        self._refresh_metadata_table()

    def _report_added_parameters(self, parameter_indices):
        if not parameter_indices:
            return
        self._populate_parameters_only()
        by_index = {parameter.index: parameter for parameter in self.project.parameters}
        for parameter_index in parameter_indices:
            parameter = by_index[parameter_index]
            self._log(
                f"Added run parameter {parameter_index} ({parameter.name})")

    def _ensure_data_loaded(self, *, force=False, index=None):
        selected = self.session.selected_run_index if index is None else index
        outcome = self.session.ensure_data_loaded(force=force, index=selected)
        state = self.session.state(selected)
        if not outcome.loaded:
            return False
        if outcome.reloaded and state.source_path is not None:
            self._log(f"Reloaded changed data source: {state.source_path.name}")
        if outcome.processing_error:
            self._log(
                f"Run {selected + 1} histograms loaded; configure processing: "
                f"{outcome.processing_error}")
        self._report_added_parameters(outcome.added_parameters)
        if selected == self.session.selected_run_index:
            self._refresh_processing_controls()
        self._refresh_runs()
        self._refresh_metadata_table()
        if state.data is not None and state.source_path is not None:
            self._log(
                f"Loaded {state.source_path.name}: {len(state.data.time_us)} fit-ready points")
        return True

    def _ensure_all_data_loaded(self, *, strict=False, plotted_only=False):
        errors = []
        for index, run in enumerate(self.project.runs):
            if plotted_only and not run.plot_enabled:
                continue
            try:
                self._ensure_data_loaded(index=index)
            except Exception as exc:
                errors.append(f"Run {index + 1} ({run.label or run.source}): {exc}")
        if strict:
            for index, run in enumerate(self.project.runs):
                if plotted_only and not run.plot_enabled:
                    continue
                if self.session.state(index).data is None:
                    message = f"Run {index + 1} ({run.label or run.source}): no fit-ready data"
                    if message not in errors:
                        errors.append(message)
        if strict and errors:
            raise ValueError("\n".join(errors))
        for error in errors:
            self._log(error)

    def _reprocess_raw_runs(self, *, strict=False):
        errors = []
        for index, run in enumerate(self.project.runs):
            if self.session.state(index).raw_data is None:
                continue
            try:
                self.session.process_raw_data(index=index)
            except Exception as exc:
                errors.append(f"Run {index + 1} ({run.label or run.source}): {exc}")
        if strict and errors:
            raise ValueError("\n".join(errors))
        for error in errors:
            self._log(error)

    def _save_project_to(self, destination: Path, *, relocating: bool) -> bool:
        """Write the native project, preserving data references across Save As."""
        try:
            destination = destination.with_suffix(".pysrfit").resolve()
            if relocating:
                # Resolve relative references against the old project directory
                # before the project acquires its new location.
                old_root = (
                    self.project.location.parent
                    if self.project.location is not None else Path.cwd()
                )
                for key in ("data_override", "spectrum_path"):
                    value = getattr(self.project, key)
                    if value and not Path(value).is_absolute():
                        setattr(self.project, key, str((old_root / value).resolve()))
                for run in self.project.runs:
                    if run.source and not Path(run.source).is_absolute():
                        run.source = str((old_root / run.source).resolve())
            self.project.save(destination)
            self._log(f"Saved {destination}")
            self.title_changed.emit()
            return True
        except Exception as exc:
            self._error(f"Save failed: {exc}")
            return False

    def save_project(self) -> bool:
        """Save in place, asking for a native path only when none exists yet."""
        try:
            self._sync()
        except Exception as exc:
            self._error(f"Cannot save: {exc}")
            return False
        location = self.project.location
        if location is None or location.suffix.lower() != ".pysrfit":
            return self.save_as_dialog(already_synced=True)
        return self._save_project_to(location, relocating=False)

    def save_as_dialog(self, *, already_synced: bool = False) -> bool:
        """Choose a new native project path and save there."""
        if not already_synced:
            try:
                self._sync()
            except Exception as exc:
                self._error(f"Cannot save: {exc}")
                return False
        default = (
            str(self.project.location.with_suffix(".pysrfit"))
            if self.project.location is not None else "project.pysrfit"
        )
        file, _ = QFileDialog.getSaveFileName(
            self, "Save native project as", default,
            "PySRFit project (*.pysrfit)")
        if not file:
            return False
        return self._save_project_to(Path(file), relocating=True)

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

    def _mapping(self, run=None):
        selected = run or self.session.current_run
        return [int(float(value)) for value in selected.tokens("map")] if selected else []

    def redraw(self, *_):
        if not hasattr(self, "canvas"):
            return
        try:
            self._sync()
            if self.project.display["t_stop"] <= self.project.display["t_start"]:
                raise ValueError("Plot time max must exceed min")
            if (not self.project.display.get("y_auto", True)
                    and self.project.display["y_max"] <= self.project.display["y_min"]):
                raise ValueError("Plot y max must exceed min")
            tmin = self.project.display["t_start"]
            tmax = self.project.display["t_stop"]
            t = np.linspace(tmin, tmax, 1400)
            self._ensure_all_data_loaded(plotted_only=True)
            # Keep plotted asymmetry synchronized with the current alpha even
            # when redraw is initiated somewhere other than Apply parameters.
            self.session.refresh_parameter_dependent_data()
            self.canvas.figure.clear()
            plotted_runs = self.session.prepared_runs(plotted_only=True)
            displayed_runs = [
                (
                    index,
                    run,
                    fit_data,
                    self.session.prepare_view_data(
                        self.view_packing_spin.value(), index=index),
                )
                for index, run, fit_data in plotted_runs
            ]
            if not plotted_runs:
                ax = self.canvas.figure.add_subplot(111)
                if not self.project.runs:
                    model = evaluate_theory(
                        self.project.theory,
                        t,
                        self.project.parameter_values(),
                        self.project.functions_text,
                        [],
                    )
                    if self.mode.currentIndex() == 1:
                        self._plot_fft_preview(ax, t, model)
                    else:
                        ax.plot(t, model, linewidth=1.8, zorder=4)
                        ax.set(
                            xlabel="Time (μs)", ylabel="Asymmetry",
                        )
                        self._apply_plot_y_range(ax)
                        ax.grid(False)
                else:
                    ax.text(
                        0.5, 0.5,
                        "No plot-selected run has fit-ready data",
                        ha="center", va="center", transform=ax.transAxes,
                    )
                    ax.set_axis_off()
            elif self.mode.currentIndex() == 1:
                ax = self.canvas.figure.add_subplot(111)
                for index, run, fit_data, data in displayed_runs:
                    self._plot_fft_dataset(
                        ax, run, fit_data, data, tmin, tmax,
                        self._plot_legend_label(index, run, data),
                        self._data_marker(index),
                    )
                ax.set(
                    xlabel=f"Frequency ({self.project.fourier['units']})",
                    ylabel=self.project.fourier["output"],
                )
                ax.set_xlim(
                    self.project.fourier["frequency_min"],
                    self.project.fourier["frequency_max"],
                )
                ax.grid(False)
                ax.legend(loc="best", fontsize="small")
            else:
                if self.residual_check.isChecked():
                    ax, residual_axis = self.canvas.figure.subplots(
                        2, 1, sharex=True,
                        gridspec_kw={"height_ratios": [3, 1]},
                    )
                else:
                    ax = self.canvas.figure.add_subplot(111)
                    residual_axis = None
                for index, run, fit_data, data in displayed_runs:
                    label = self._plot_legend_label(index, run, data)
                    marker = self._data_marker(index)
                    color = self._plot_dataset(
                        ax, run, fit_data, data, tmin, tmax, label, marker)
                    if residual_axis is not None:
                        self._plot_residual(
                            residual_axis, run, fit_data, data,
                            tmin, tmax, color, marker)
                kinds = {"Counts" if isinstance(data, CountSpectrum) else "Asymmetry"
                         for _index, _run, _fit_data, data in displayed_runs}
                ax.set(
                    ylabel=next(iter(kinds)) if len(kinds) == 1 else "Signal",
                )
                self._apply_plot_y_range(ax)
                ax.grid(False)
                ax.legend(loc="best", fontsize="small")
                if residual_axis is not None:
                    residual_axis.axhline(0, color="grey", linewidth=0.8)
                    residual_axis.set(xlabel="Time (μs)", ylabel="Residual")
                    residual_axis.grid(False)
                else:
                    ax.set_xlabel("Time (μs)")
            self.canvas.draw_idle()
        except Exception as exc:
            self._log(f"Plot/model error: {exc}")

    @staticmethod
    def _data_marker(index):
        return DATA_MARKERS[index % len(DATA_MARKERS)]

    @staticmethod
    def _plot_legend_label(index, run, data):
        metadata = dict(getattr(data, "metadata", {}) or {})
        metadata.update(run.file_metadata or {})
        run_number = infer_run_number(run.source, metadata)
        return str(run_number) if run_number is not None else f"Run {index + 1}"

    def _evaluate_plot_model(self, run, fit_data, plot_data, time):
        parameter_values = None
        if isinstance(fit_data, CountSpectrum) and isinstance(plot_data, CountSpectrum):
            fit_packing = float(fit_data.metadata.get("packing", 1))
            view_packing = float(plot_data.metadata.get("packing", fit_packing))
            if fit_packing > 0 and view_packing != fit_packing:
                scale = view_packing / fit_packing
                parameter_values = self.project.parameter_values()
                for key in ("norm", "backgr.fit"):
                    tokens = run.tokens(key)
                    if not tokens:
                        continue
                    parameter_index = int(float(tokens[0]))
                    if parameter_index in parameter_values:
                        parameter_values[parameter_index] *= scale
        return evaluate_data_model(
            self.project,
            plot_data,
            time,
            parameter_values,
            run=run,
        )

    def _plot_dataset(
        self, ax, run, fit_data, data, tmin, tmax, label, marker,
    ):
        selected = (data.time_us >= tmin) & (data.time_us <= tmax)
        indices = np.flatnonzero(selected)
        indices = indices[::max(1, len(indices) // 1500)]
        artists = ax.errorbar(
            data.time_us[indices], data.values[indices], yerr=data.error[indices],
            fmt=marker, markersize=7, markeredgecolor="black",
            markeredgewidth=0.7, linewidth=0.5, alpha=0.85,
            label=label, zorder=2,
        )
        color = artists[0].get_color()
        time = np.linspace(tmin, tmax, 1400)
        model = self._evaluate_plot_model(run, fit_data, data, time)
        ax.plot(
            time, model, linewidth=1.8, color=color,
            label="_nolegend_", zorder=4)
        return color

    def _plot_residual(
        self, ax, run, fit_data, data, tmin, tmax, color, marker,
    ):
        selected = (data.time_us >= tmin) & (data.time_us <= tmax)
        indices = np.flatnonzero(selected)
        indices = indices[::max(1, len(indices) // 1500)]
        time = data.time_us[indices]
        model = self._evaluate_plot_model(run, fit_data, data, time)
        ax.errorbar(
            time, data.values[indices] - model, yerr=data.error[indices],
            fmt=marker, color=color, ecolor=color, markersize=6,
            markeredgecolor="black", markeredgewidth=0.6,
            linewidth=0.4, alpha=0.75, label="_nolegend_",
        )

    def _apply_plot_y_range(self, ax):
        if not self.project.display.get("y_auto", True):
            ax.set_ylim(
                float(self.project.display["y_min"]),
                float(self.project.display["y_max"]),
            )

    def _plot_fft_dataset(
        self, ax, run, fit_data, data, tmin, tmax, label, marker,
    ):
        valid = (data.time_us >= tmin) & (data.time_us <= tmax)
        tx = data.time_us[valid]
        yy = data.values[valid]
        data_artist = self._plot_fft_values(
            ax,
            tx,
            yy,
            label,
            linestyle="none",
            marker=marker,
            markersize=7,
            markeredgecolor="black",
            markeredgewidth=0.7,
            alpha=0.85,
            zorder=2,
        )
        model = self._evaluate_plot_model(run, fit_data, data, tx)
        self._plot_fft_values(
            ax,
            tx,
            model,
            "_nolegend_",
            color=data_artist.get_color(),
            linestyle="-",
            linewidth=1.8,
            zorder=4,
        )

    def _plot_fft_preview(self, ax, time, model):
        self._plot_fft_values(ax, time, model, "Model")
        ax.set(
            xlabel=f"Frequency ({self.project.fourier['units']})",
            ylabel=self.project.fourier["output"],
        )
        ax.set_xlim(
            self.project.fourier["frequency_min"],
            self.project.fourier["frequency_max"],
        )
        ax.grid(False)

    def _plot_fft_values(self, ax, tx, yy, label, **plot_options):
        settings = self.project.fourier
        if settings["frequency_max"] <= settings["frequency_min"]:
            raise ValueError("Fourier frequency max must exceed min")
        frequencies, transformed = fourier_transform(
            tx,
            yy,
            window=settings["window"],
            subtract_mean=settings["subtract_mean"],
            zero_padding=settings["zero_padding"],
            phase_deg=settings["phase_deg"],
            output=settings["output"],
            normalize=settings["normalize"],
            units=settings["units"],
        )
        visible = (
            (frequencies >= settings["frequency_min"])
            & (frequencies <= settings["frequency_max"])
        )
        artist, = ax.plot(
            frequencies[visible],
            transformed[visible],
            label=label,
            **plot_options,
        )
        return artist

    def start_sequential_fit(self):
        if self._batch_thread is not None:
            self._log("A sequential fit is already running")
            return
        try:
            self._sync()
            if len(self.project.runs) != 1:
                raise ValueError("Sequential fitting requires exactly one run")
            if self.project.location is None:
                raise ValueError("Save the reference project before starting a sequential fit")
        except Exception as exc:
            self._error(f"Cannot start sequential fit: {exc}")
            return

        dialog = SequentialFitDialog(self.project, self)
        if dialog.exec_() != dialog.Accepted:
            return
        try:
            run_numbers = dialog.run_numbers
            targets, _result_path = sequential_plan(self.project, run_numbers)
            existing = sequential_existing_outputs(self.project, run_numbers)
        except Exception as exc:
            self._error(f"Cannot start sequential fit: {exc}")
            return

        overwrite = False
        if existing:
            names = "\n".join(f"• {path.name}" for path in existing[:12])
            if len(existing) > 12:
                names += f"\n• … and {len(existing) - 12} more"
            answer = QMessageBox.question(
                self,
                "Replace existing sequential outputs?",
                "The following files already exist:\n\n"
                f"{names}\n\nReplace them?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
            overwrite = True

        reference = copy.deepcopy(self.project)
        reference.location = self.project.location
        progress = QProgressDialog(
            "Preparing sequential fit…",
            "Cancel",
            0,
            len(targets),
            self,
        )
        progress.setWindowTitle("Sequential fit")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)

        thread = QThread(self)
        worker = SequentialFitWorker(
            reference,
            run_numbers,
            dialog.axis,
            dialog.custom_field,
            dialog.chain_check.isChecked(),
            dialog.continue_check.isChecked(),
            overwrite,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._sequential_progress_changed)
        worker.completed.connect(self._sequential_fit_completed)
        worker.failed.connect(self._sequential_fit_failed)
        worker.completed.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._sequential_fit_cleaned_up)
        progress.canceled.connect(lambda: worker.cancel())

        self._batch_thread = thread
        self._batch_worker = worker
        self._batch_progress = progress
        self.sequential_fit_button.setEnabled(False)
        progress.show()
        thread.start()

    def _sequential_progress_changed(self, current, total, message):
        progress = self._batch_progress
        if progress is None:
            return
        # setValue() may process queued completion/cleanup events. Keep a local
        # reference and update the label first so reentrant cleanup cannot turn
        # the following access into a None dereference.
        progress.setLabelText(message)
        progress.setMaximum(max(1, total))
        progress.setValue(min(current, max(1, total)))

    def _sequential_fit_completed(self, summary):
        progress = self._batch_progress
        if progress is not None:
            progress.setLabelText("Sequential fit complete")
            progress.setValue(progress.maximum())
            progress.close()
        failed = [outcome for outcome in summary.outcomes if not outcome.fit_valid]
        omitted = [
            outcome for outcome in summary.outcomes
            if outcome.fit_valid and outcome.message
        ]
        for outcome in summary.outcomes:
            run = outcome.target.run_number
            label = outcome.target.source_path.name if run is None else f"run {run}"
            if outcome.fit_valid:
                message = f"Sequential fit saved {label}: {outcome.target.project_path.name}"
                if outcome.message:
                    message += f" ({outcome.message})"
            else:
                message = f"Sequential fit failed for {label}: {outcome.message}"
            self._log(message)
        state = "cancelled" if summary.cancelled else "complete"
        QMessageBox.information(
            self,
            "Sequential fit",
            f"Sequential fit {state}.\n\n"
            f"Table rows: {summary.successful_rows}\n"
            f"Failed fits: {len(failed)}\n"
            f"Fits omitted for missing scan axis: {len(omitted)}\n\n"
            f"Results: {summary.result_path}",
        )

    def _sequential_fit_failed(self, message):
        progress = self._batch_progress
        if progress is not None:
            progress.close()
        self._error(f"Sequential fit failed: {message}")

    def _sequential_fit_cleaned_up(self):
        self._batch_thread = None
        self._batch_worker = None
        self._batch_progress = None
        self.sequential_fit_button.setEnabled(True)

    def fit(self, *_):
        try:
            self._sync()
            self._ensure_all_data_loaded(strict=True)
        except Exception as exc:
            self._error(f"Cannot start fit: {exc}")
            return
        prepared = self.session.prepared_runs()
        if not prepared:
            self._error("Cannot fit: no fit-ready runs are attached.")
            return
        before_fit_text = self.parameters_edit.toPlainText()
        try:
            result = fit_global(
                self.project,
                [(run, data) for _index, run, data in prepared],
            )
            if result.automatically_fixed:
                by_index = {
                    parameter.index: parameter.name
                    for parameter in self.project.parameters
                }
                descriptions = ", ".join(
                    f"{index} ({by_index[index]})"
                    for index in result.automatically_fixed)
                self._log(
                    "Automatically fixed unused parameters for this fit: "
                    f"{descriptions}")
            if not result.valid:
                self.project.last_fit = FitRecord(
                    valid=False,
                    chi2=result.chi2,
                    ndf=result.ndf,
                    values=result.values,
                    errors=result.errors,
                    run_chi2=result.run_chi2,
                    run_points=result.run_points,
                    message="MINUIT did not converge to a valid minimum",
                )
                self._log("MINUIT did not converge to a valid minimum; parameters NOT applied")
                if not self.save_project():
                    self._log("The failed fit record was not saved")
                return
            for p in self.project.parameters:
                p.value = result.values[p.index]
                if not p.fixed and p.index not in result.automatically_fixed:
                    p.step = result.errors[p.index]
            self._populate_parameters_only()
            self._sync_parameters()
            self.session.refresh_parameter_dependent_data()
            self.project.last_fit = FitRecord(
                valid=True,
                chi2=result.chi2,
                ndf=result.ndf,
                values=result.values,
                errors=result.errors,
                run_chi2=result.run_chi2,
                run_points=result.run_points,
            )
            self._remember_fit_parameters(
                before_fit_text, self.parameters_edit.toPlainText())
            self._log(f"MINUIT2: χ²={result.chi2:.5g}, NDF={result.ndf}, χ²/NDF={result.chi2/result.ndf:.5g}")
            for (index, run, _data), chi2, points in zip(
                    prepared, result.run_chi2, result.run_points):
                self._log(
                    f"  Run {index + 1} {run.label or Path(run.source).name}: "
                    f"χ²={chi2:.5g} for {points} points")
            self.redraw()
            if not self.save_project():
                self._log("The fit result was not saved")
        except Exception as exc:
            self._error(f"Fit failed: {exc}")

    def export_plot(self):
        file, _ = QFileDialog.getSaveFileName(self, "Export figure", "pysrfit_plot.png", "PNG (*.png)")
        if file:
            self.canvas.figure.savefig(file, dpi=250)
            self._log(f"Saved plot: {file}")

    def _error(self, message):
        self._log(message)
        QMessageBox.warning(self, "PySRFit", message)
