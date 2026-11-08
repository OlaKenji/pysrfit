from pathlib import Path

import pytest

from pysrfit.models import FitRecord, Parameter, Project, RunConfig
from pysrfit.results import (
    AXIS_CUSTOM,
    AXIS_FIELD,
    AXIS_INDEX,
    AXIS_TEMPERATURE,
    collect_project_results,
    parameter_schema,
    result_row,
    scan_axis_value,
    write_results_table,
)


def fitted_project(axis_metadata, value=0.2):
    return Project(
        parameters=[Parameter(1, "asymmetry", value, 0.01)],
        runs=[RunConfig(file_metadata=axis_metadata)],
        last_fit=FitRecord(
            valid=True,
            chi2=10.0,
            ndf=8,
            values={1: value},
            errors={1: 0.005},
            run_chi2=[10.0],
            run_points=[10],
        ),
    )


def test_fit_record_round_trips_integer_parameter_keys(tmp_path):
    path = tmp_path / "fit.pysrfit"
    project = fitted_project({"temperature": "5 K"})
    project.save(path)

    loaded = Project.load(path)

    assert loaded.last_fit.valid
    assert loaded.last_fit.values == {1: 0.2}
    assert loaded.last_fit.errors == {1: 0.005}
    assert loaded.last_fit.chi2 == 10.0
    assert loaded.last_fit.ndf == 8


def test_scan_axis_converts_temperature_and_field_units():
    assert scan_axis_value(
        fitted_project({"temperature": "250 mK"}), AXIS_TEMPERATURE) == 0.25
    assert scan_axis_value(
        fitted_project({"field": "1000 G"}), AXIS_FIELD) == pytest.approx(0.1)
    assert scan_axis_value(
        fitted_project({
            "temperature": "20.00 +- 0.04 K; SP: 20; sensor=LTC21",
        }),
        AXIS_TEMPERATURE,
    ) == 20.0
    assert scan_axis_value(
        fitted_project({"field": "5 +/- 0.1 G; automatic ZFC"}),
        AXIS_FIELD,
    ) == pytest.approx(5e-4)
    assert scan_axis_value(
        fitted_project({"scan_position": "3.5 mm"}),
        AXIS_CUSTOM,
        custom_field="scan_position",
    ) == 3.5
    assert scan_axis_value(
        fitted_project({}), AXIS_INDEX, sequence_index=4) == 4


def test_collection_sorts_axis_and_writes_compact_tsv(tmp_path):
    warm = fitted_project({"temperature": "10 K"}, 0.1)
    cold = fitted_project({"temperature": "5 K"}, 0.2)
    warm_path = tmp_path / "warm.pysrfit"
    cold_path = tmp_path / "cold.pysrfit"
    warm.save(warm_path)
    cold.save(cold_path)

    schema, rows = collect_project_results(
        [warm_path, cold_path], AXIS_TEMPERATURE)
    output = write_results_table(
        tmp_path / "results.csv", AXIS_TEMPERATURE, schema, rows)

    assert [row.axis_value for row in rows] == [5.0, 10.0]
    assert output.read_text().splitlines() == [
        "temperature_K,p1_asymmetry,p1_asymmetry_err",
        "5,0.2,0.005",
        "10,0.1,0.005",
    ]


def test_collection_rejects_parameter_schema_mismatch(tmp_path):
    first = fitted_project({"temperature": 1})
    second = fitted_project({"temperature": 2})
    second.parameters[0].name = "different"
    first_path = tmp_path / "first.pysrfit"
    second_path = tmp_path / "second.pysrfit"
    first.save(first_path)
    second.save(second_path)

    with pytest.raises(ValueError, match="does not match"):
        collect_project_results([first_path, second_path], AXIS_TEMPERATURE)


def test_collection_without_explicit_record_uses_parameter_values_and_steps(tmp_path):
    project = Project(
        parameters=[Parameter(1, "rate", 0.25, 0.015)],
        runs=[RunConfig(file_metadata={"temperature": "7 K"})],
    )
    path = tmp_path / "existing_fit.pysrfit"
    project.save(path)

    schema, rows = collect_project_results([path], AXIS_TEMPERATURE)

    assert schema[0].name == "rate"
    assert rows[0].values == {1: 0.25}
    assert rows[0].errors == {1: 0.015}
