"""Regression checks for the GUI application object."""

from mysr.app import MysrApplication, main


def test_application_starts_without_creating_qt_objects():
    application = MysrApplication(["mysr"])
    assert application.argv == ["mysr"]
    assert application.qt_application is None
    assert application.main_window is None


def test_application_owns_window_lifecycle():
    events = []

    class FakeQtApplication:
        def exec_(self):
            events.append("exec")
            return 7

    class FakeWindow:
        def show(self):
            events.append("show")

    application = MysrApplication(["mysr"])
    application.create_qt_application = lambda: FakeQtApplication()
    application.create_main_window = lambda: FakeWindow()
    assert application.run() == 7
    assert events == ["show", "exec"]


def test_gui_entry_point_delegates_to_application(monkeypatch):
    monkeypatch.setattr(MysrApplication, "run", lambda self: 11)
    assert main() == 11

