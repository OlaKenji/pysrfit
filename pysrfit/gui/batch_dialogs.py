"""Dialogs and worker objects for sequential fitting and result collection."""
from __future__ import annotations

from pathlib import Path
import threading

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QVBoxLayout,
)

from ..batch import parse_run_numbers, run_sequential_fit
from ..models import Project
from ..results import AXIS_CHOICES, AXIS_CUSTOM


class _AxisDialog(QDialog):
    def _add_axis_controls(self, layout: QVBoxLayout) -> None:
        form = QFormLayout()
        self.axis_combo = QComboBox()
        for label, value in AXIS_CHOICES:
            self.axis_combo.addItem(label, value)
        form.addRow("Scan axis:", self.axis_combo)
        self.custom_axis_edit = QLineEdit()
        self.custom_axis_edit.setPlaceholderText("metadata field name")
        self.custom_axis_label = QLabel("Custom field:")
        form.addRow(self.custom_axis_label, self.custom_axis_edit)
        layout.addLayout(form)
        self.axis_combo.currentIndexChanged.connect(self._axis_changed)
        self._axis_changed()

    def _axis_changed(self, *_):
        visible = self.axis == AXIS_CUSTOM
        self.custom_axis_label.setVisible(visible)
        self.custom_axis_edit.setVisible(visible)

    @property
    def axis(self) -> str:
        return str(self.axis_combo.currentData())

    @property
    def custom_field(self) -> str:
        return self.custom_axis_edit.text().strip()

    def _validate_axis(self) -> bool:
        if self.axis == AXIS_CUSTOM and not self.custom_field:
            QMessageBox.warning(
                self, "Missing metadata field", "Enter a custom metadata field name.")
            return False
        return True


class SequentialFitDialog(_AxisDialog):
    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self.project = project
        self.setWindowTitle("Sequential fit")
        self.resize(620, 330)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "The current single-run project is the reference and is always fitted first. "
            "All projects and the result table are written beside the reference project."))
        location = project.location.parent if project.location is not None else None
        layout.addWidget(QLabel(
            f"Output directory: {location if location is not None else 'save the project first'}"))
        form = QFormLayout()
        self.runs_edit = QLineEdit()
        self.runs_edit.setPlaceholderText("e.g. 3092-3100, 3105, 3110-3120:2")
        form.addRow("Additional runs:", self.runs_edit)
        layout.addLayout(form)
        layout.addWidget(QLabel(
            "The reference run is included automatically. Missing or invalid runs are "
            "saved with a failed fit record and omitted from the result table."))
        self._add_axis_controls(layout)
        self.chain_check = QCheckBox(
            "Chain fit (use the last successful values as the next starting values)")
        self.chain_check.setChecked(True)
        layout.addWidget(self.chain_check)
        self.continue_check = QCheckBox("Continue after missing data or failed fits")
        self.continue_check.setChecked(True)
        layout.addWidget(self.continue_check)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Start sequential fit")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def run_numbers(self) -> list[int]:
        return parse_run_numbers(self.runs_edit.text())

    def accept(self):
        try:
            self.run_numbers
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid run numbers", str(exc))
            return
        if not self._validate_axis():
            return
        super().accept()


class CollectionOptionsDialog(_AxisDialog):
    def __init__(self, paths: list[Path], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Collect fit results")
        self.resize(620, 360)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Projects must have matching parameter numbers/names; comments and editor "
            "layout are ignored."))
        files = QPlainTextEdit()
        files.setReadOnly(True)
        files.setPlainText("\n".join(str(path) for path in paths))
        layout.addWidget(files, 1)
        self._add_axis_controls(layout)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Export table")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def accept(self):
        if self._validate_axis():
            super().accept()


class SequentialFitWorker(QObject):
    progress = pyqtSignal(int, int, str)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        project: Project,
        run_numbers: list[int],
        axis: str,
        custom_field: str,
        chain_fit: bool,
        continue_on_error: bool,
        overwrite: bool,
    ):
        super().__init__()
        self.project = project
        self.run_numbers = run_numbers
        self.axis = axis
        self.custom_field = custom_field
        self.chain_fit = chain_fit
        self.continue_on_error = continue_on_error
        self.overwrite = overwrite
        self._cancelled = threading.Event()

    def cancel(self):
        self._cancelled.set()

    @pyqtSlot()
    def run(self):
        try:
            summary = run_sequential_fit(
                self.project,
                self.run_numbers,
                self.axis,
                custom_field=self.custom_field,
                chain_fit=self.chain_fit,
                continue_on_error=self.continue_on_error,
                overwrite=self.overwrite,
                progress=lambda current, total, message: self.progress.emit(
                    current, total, message),
                cancelled=self._cancelled.is_set,
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.completed.emit(summary)
