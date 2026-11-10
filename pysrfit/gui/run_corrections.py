"""Interactive raw-histogram editor for t0 and fit/background ranges."""
from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QVBoxLayout,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure
import numpy as np

from ..corrections import RunCorrectionState
from ..io.base import HistogramRun
from ..models import RunConfig


class RunCorrectionsDialog(QDialog):
    """Select musrfit-style RUN corrections directly on raw histograms."""

    KEY_ALIASES = {
        "t": "t",
        "d": "d",
        "D": "D",
        "shift+d": "D",
        "b": "b",
        "B": "B",
        "shift+b": "B",
    }

    def __init__(self, data: HistogramRun, run: RunConfig, parent=None):
        super().__init__(parent)
        self.state = RunCorrectionState(data, run)
        self._cursor_channel = None
        self._cursor_line = None
        self._blit_background = None
        self._axes = None
        self.setWindowTitle("Visual run corrections")
        self.resize(1050, 700)

        layout = QVBoxLayout(self)
        if self.state.background_editable:
            help_text = (
                "Move the pointer over the raw histogram and press: "
                "t = t0, d/D = data start/end, and b/B = background "
                "start/end. Choose below whether edits affect the current "
                "histogram/group or every selected histogram. Uppercase D and "
                "B mean Shift+D and Shift+B.")
        else:
            help_text = (
                "Move the pointer over the raw histogram and press: "
                "t = t0 and d/D = data start/end. Choose below whether edits "
                "affect the current histogram/group or every selected histogram. "
                "The constant background is fitted, "
                "so background-range markers are not used.")
        help_label = QLabel(help_text)
        help_label.setWordWrap(True)
        layout.addWidget(help_label)

        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("Histogram:"))
        self.target_combo = QComboBox()
        for target in self.state.targets:
            self.target_combo.addItem(target.label)
        self.target_combo.currentIndexChanged.connect(self._target_changed)
        selector_row.addWidget(self.target_combo, 1)
        selector_row.addWidget(QLabel("Apply edits to:"))
        self.scope_combo = QComboBox()
        self.scope_combo.addItem("Current histogram / group", False)
        self.scope_combo.addItem("All selected histograms", True)
        default_scope = self.scope_combo.findData(self.state.default_apply_all)
        self.scope_combo.setCurrentIndex(max(0, default_scope))
        selector_row.addWidget(self.scope_combo)
        layout.addLayout(selector_row)

        self.figure = Figure(figsize=(10, 6), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        layout.addWidget(self.canvas, 1)

        self.status_label = QLabel()
        layout.addWidget(self.status_label)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.canvas.mpl_connect("motion_notify_event", self._pointer_moved)
        self.canvas.mpl_connect("button_press_event", self._canvas_clicked)
        self.canvas.mpl_connect("key_press_event", self._key_pressed)
        self.canvas.mpl_connect("draw_event", self._capture_blit_background)
        self._draw()

    @property
    def target_index(self) -> int:
        return max(0, self.target_combo.currentIndex())

    def _target_changed(self, _index):
        self._cursor_channel = None
        self._draw()

    def _draw(self, *, preserve_limits=False):
        old_limits = self._axes.get_xlim() if preserve_limits and self._axes is not None else None
        self._blit_background = None
        self.figure.clear()
        self._axes = self.figure.add_subplot(111)
        counts = self.state.counts(self.target_index)
        channels = np.arange(len(counts))
        self._axes.plot(channels, counts, color="black", linewidth=0.8, drawstyle="steps-mid")
        markers = self.state.marker_values(self.target_index)

        data_range = markers["data"]
        if None not in data_range:
            self._axes.axvspan(data_range[0], data_range[1], color="tab:green", alpha=0.18, label="Data range")
        background = markers["background"]
        if self.state.background_editable and None not in background:
            self._axes.axvspan(background[0], background[1], color="tab:blue", alpha=0.18, label="Background")
        if markers["t0"] is not None:
            self._axes.axvline(markers["t0"], color="tab:red", linewidth=1.5, label="t0")
        cursor_x = 0 if self._cursor_channel is None else self._cursor_channel
        self._cursor_line = self._axes.axvline(
            cursor_x, color="0.45", linestyle="--", linewidth=0.8,
            animated=True, visible=self._cursor_channel is not None)
        self._axes.set(
            xlabel="Raw histogram channel",
            ylabel="Counts",
            title=self.state.targets[self.target_index].label,
        )
        self._axes.grid(alpha=0.15)
        handles, labels = self._axes.get_legend_handles_labels()
        if handles:
            self._axes.legend(handles, labels, loc="best")
        if old_limits is not None:
            self._axes.set_xlim(old_limits)
        self._update_status()
        # A full draw is needed only when the histogram or correction markers
        # change. Pointer motion restores this cached image and blits the
        # animated cursor line on top of it.
        self.canvas.draw()
        self.canvas.setFocus()

    def _capture_blit_background(self, event):
        if event.canvas is not self.canvas or self._axes is None:
            return
        self._blit_background = self.canvas.copy_from_bbox(self._axes.bbox)
        self._blit_cursor()

    def _blit_cursor(self):
        if self._blit_background is None or self._axes is None or self._cursor_line is None:
            return
        self.canvas.restore_region(self._blit_background)
        if self._cursor_line.get_visible():
            self._axes.draw_artist(self._cursor_line)
        self.canvas.blit(self._axes.bbox)

    def _update_status(self, message=None):
        markers = self.state.marker_values(self.target_index)
        cursor = "—" if self._cursor_channel is None else str(self._cursor_channel)
        summary = f"Pointer: {cursor} | t0: {markers['t0']} | data: {markers['data']}"
        if self.state.background_editable:
            summary += f" | background: {markers['background']}"
        self.status_label.setText(f"{message}   {summary}" if message else summary)

    def _pointer_moved(self, event):
        if event.inaxes is not self._axes or event.xdata is None:
            return
        channel = int(round(event.xdata))
        channel = min(max(channel, 0), self.state.data.counts.shape[1] - 1)
        if channel == self._cursor_channel:
            return
        self._cursor_channel = channel
        self._cursor_line.set_xdata([channel, channel])
        self._cursor_line.set_visible(True)
        self._update_status()
        self._blit_cursor()

    def _canvas_clicked(self, event):
        if event.inaxes is self._axes:
            self.canvas.setFocus()

    def _key_pressed(self, event):
        key = self.KEY_ALIASES.get(event.key)
        if key is None:
            return
        if event.xdata is not None and event.inaxes is self._axes:
            self._cursor_channel = int(round(event.xdata))
        if self._cursor_channel is None:
            self._update_status("Move the pointer over the histogram first.")
            return
        try:
            self.state.set_marker(
                self.target_index,
                key,
                self._cursor_channel,
                apply_all=bool(self.scope_combo.currentData()),
            )
        except ValueError as exc:
            self._update_status(str(exc))
            return
        self._draw(preserve_limits=True)

    def accept(self):
        try:
            self.state.validate()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid run corrections", str(exc))
            return
        super().accept()

    def apply_to(self, run: RunConfig) -> None:
        self.state.apply_to(run)
