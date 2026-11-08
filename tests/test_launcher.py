"""Regression checks for the GUI application object."""

from pysrfit.app import PySRFitApplication, main


def test_application_starts_without_creating_qt_objects():
    application = PySRFitApplication(["pysrfit"])
    assert application.argv == ["pysrfit"]
    assert application.qt_application is None
    assert application.main_window is None


def test_application_owns_window_lifecycle():
    events = []

    class FakeQtApplication:
        def exec_(self):
            events.append("exec")
            return 7

    class FakeWindow:
        def showMaximized(self):
            events.append("show-maximized")

    application = PySRFitApplication(["pysrfit"])
    application.create_qt_application = lambda: FakeQtApplication()
    application.create_main_window = lambda: FakeWindow()
    assert application.run() == 7
    assert events == ["show-maximized", "exec"]


def test_gui_entry_point_delegates_to_application(monkeypatch):
    monkeypatch.setattr(PySRFitApplication, "run", lambda self: 11)
    assert main() == 11

