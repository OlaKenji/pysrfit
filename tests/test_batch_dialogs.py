import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication

from pysrfit.gui.batch_dialogs import CollectionOptionsDialog, SequentialFitDialog
from pysrfit.models import Project
from pysrfit.results import AXIS_CUSTOM


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


def test_sequential_dialog_defaults_to_chain_fit(application, tmp_path):
    project = Project()
    project.location = tmp_path / "reference.pysrfit"
    dialog = SequentialFitDialog(project)
    dialog.runs_edit.setText("3-5, 8")

    assert dialog.run_numbers == [3, 4, 5, 8]
    assert dialog.chain_check.isChecked()
    assert dialog.continue_check.isChecked()
    assert dialog.custom_axis_edit.isHidden()
    dialog.close()


def test_collection_dialog_shows_custom_metadata_field(application, tmp_path):
    dialog = CollectionOptionsDialog([tmp_path / "fit.pysrfit"])
    dialog.axis_combo.setCurrentIndex(dialog.axis_combo.findData(AXIS_CUSTOM))

    assert dialog.axis == AXIS_CUSTOM
    assert not dialog.custom_axis_edit.isHidden()
    dialog.close()
