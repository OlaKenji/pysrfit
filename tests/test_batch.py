from pathlib import Path

import pytest

from pysrfit.batch import parse_run_numbers, run_sequential_fit, sequential_plan
from pysrfit.core.fit import FitResult
from pysrfit.models import Parameter, Project, RunConfig
from pysrfit.results import AXIS_INDEX


def test_run_number_expression_supports_ranges_steps_and_descending():
    assert parse_run_numbers("3-5, 8, 12-16:2") == [3, 4, 5, 8, 12, 14, 16]
    assert parse_run_numbers("5-3") == [5, 4, 3]
    assert parse_run_numbers("3,3,4") == [3, 4]
    with pytest.raises(ValueError, match="Invalid"):
        parse_run_numbers("3-nope")


def test_sequential_plan_includes_reference_and_uses_reference_directory(tmp_path):
    data = tmp_path / "gps_2025_003092.root"
    data.write_text("data")
    reference_path = tmp_path / "scan.pysrfit"
    project = Project(
        runs=[RunConfig(
            source=str(data),
            file_metadata={"run_number": 3092},
        )],
    )
    project.save(reference_path)

    targets, result_path = sequential_plan(project, [3092, 3093, 3095])

    assert [target.run_number for target in targets] == [3092, 3093, 3095]
    assert targets[0].is_reference
    assert [target.project_path.name for target in targets] == [
        "gps_2025_003092.pysrfit",
        "gps_2025_003093.pysrfit",
        "gps_2025_003095.pysrfit",
    ]
    assert result_path == tmp_path / "scan_results.csv"


def test_sequential_plan_requires_saved_single_run_reference(tmp_path):
    with pytest.raises(ValueError, match="exactly one run"):
        sequential_plan(Project(), [])
    data = tmp_path / "run_1.dat"
    data.write_text("data")
    project = Project(runs=[RunConfig(source=str(data))])
    with pytest.raises(ValueError, match="Save the reference"):
        sequential_plan(project, [])


def test_sequential_fit_writes_every_project_chains_and_exports_table(
        tmp_path, monkeypatch):
    first = tmp_path / "run_0001.dat"
    third = tmp_path / "run_0003.dat"
    spectrum = "0 0.2 0.01\n1 0.1 0.01\n"
    first.write_text(spectrum)
    third.write_text(spectrum)
    reference_path = tmp_path / "template.pysrfit"
    project = Project(
        parameters=[Parameter(1, "asymmetry", 0.0, 0.1)],
        theory="asymmetry 1",
        runs=[RunConfig(
            source=str(first),
            label=first.stem,
            file_metadata={"run_number": 1},
        )],
    )
    project.save(reference_path)
    starts = []

    def fake_fit(fitted_project, datasets):
        assert len(datasets) == 1
        start = fitted_project.parameters[0].value
        starts.append(start)
        value = start + 1.0
        return FitResult(
            values={1: value},
            errors={1: 0.05},
            chi2=value * 10,
            ndf=1,
            valid=True,
            run_chi2=[value * 10],
            run_points=[2],
        )

    monkeypatch.setattr("pysrfit.batch.fit_global", fake_fit)

    summary = run_sequential_fit(
        project, [3], AXIS_INDEX, chain_fit=True)

    assert starts == [0.0, 1.0]
    assert summary.successful_rows == 2
    assert not summary.cancelled
    first_project = Project.load(tmp_path / "run_0001.pysrfit")
    third_project = Project.load(tmp_path / "run_0003.pysrfit")
    assert first_project.last_fit.values == {1: 1.0}
    assert first_project.last_fit.chi2 == 10.0
    assert third_project.last_fit.values == {1: 2.0}
    assert summary.result_path.read_text().splitlines() == [
        "scan_index,p1_asymmetry,p1_asymmetry_err",
        "1,1,0.05",
        "2,2,0.05",
    ]
