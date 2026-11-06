"""GUI application lifecycle for mysr."""
from __future__ import annotations

import sys


class MysrApplication:
    """Own the Qt application and its main window."""

    def __init__(self, argv=None):
        self.argv = list(sys.argv if argv is None else argv)
        self.qt_application = None
        self.main_window = None

    def create_qt_application(self):
        from PyQt5.QtWidgets import QApplication

        self.qt_application = QApplication.instance() or QApplication(self.argv)
        return self.qt_application

    def create_main_window(self):
        from .gui.main_window import MainWindow

        self.main_window = MainWindow()
        return self.main_window

    def run(self) -> int:
        application = self.create_qt_application()
        window = self.create_main_window()
        window.show()
        return int(application.exec_())


def main() -> int:
    """GUI entry point used by the installed application launcher."""
    return MysrApplication().run()

