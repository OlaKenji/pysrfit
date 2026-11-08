"""Shared scan-axis extraction and portable tabular fit-result export."""
from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any, Iterable

from .models import Project


AXIS_TEMPERATURE = "temperature"
AXIS_FIELD = "field"
AXIS_INDEX = "index"
AXIS_CUSTOM = "custom"
AXIS_CHOICES = (
    ("Temperature (K)", AXIS_TEMPERATURE),
    ("Magnetic field (T)", AXIS_FIELD),
    ("Sequence / scan index", AXIS_INDEX),
    ("Custom metadata field", AXIS_CUSTOM),
)

_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_UNCERTAINTY = re.compile(
    rf"^\s*(?:(?:\+/-|\+-|±)\s*{_NUMBER.pattern}|\(\d+\))\s*")
_UNIT = re.compile(r"^([A-Za-zµμ°]+)")


@dataclass(frozen=True)
class ParameterSpec:
    index: int
    name: str


@dataclass(frozen=True)
class ResultRow:
    axis_value: float
    values: dict[int, float]
    errors: dict[int, float]
    project_path: Path | None = None


def parameter_schema(project: Project) -> tuple[ParameterSpec, ...]:
    return tuple(
        ParameterSpec(parameter.index, parameter.name)
        for parameter in sorted(project.parameters, key=lambda item: item.index)
    )


def _metadata_items(metadata: dict[str, Any], prefix: str = ""):
    for key, value in metadata.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            yield from _metadata_items(value, name)
        else:
            yield name, value


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _metadata_value(project: Project, candidates: Iterable[str]) -> Any:
    if not project.runs:
        raise ValueError("Project contains no run metadata")
    run = project.runs[0]
    items = list(_metadata_items(run.file_metadata))
    items.extend(_metadata_items(run.user_metadata))
    wanted = {_normalized(candidate) for candidate in candidates}
    for key, value in items:
        leaf = key.rsplit(".", 1)[-1]
        if _normalized(key) in wanted or _normalized(leaf) in wanted:
            return value
    raise ValueError(
        "Metadata field not found: " + ", ".join(str(value) for value in candidates))


def _quantity(value: Any) -> tuple[float, str]:
    if isinstance(value, bool):
        raise ValueError(f"Expected a numeric metadata value, got {value!r}")
    if isinstance(value, (int, float)):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"Metadata value is not finite: {value!r}")
        return numeric, ""
    text = str(value).strip()
    match = _NUMBER.search(text)
    if match is None:
        raise ValueError(f"Expected a numeric metadata value, got {value!r}")
    numeric = float(match.group())
    remainder = text[match.end():]
    uncertainty = _UNCERTAINTY.match(remainder)
    if uncertainty is not None:
        remainder = remainder[uncertainty.end():]
    remainder = remainder.strip()
    unit_match = _UNIT.match(remainder)
    unit = (
        unit_match.group(1).replace("μ", "u").replace("µ", "u").casefold()
        if unit_match is not None else ""
    )
    return numeric, unit


def axis_header(axis: str, custom_field: str = "") -> str:
    if axis == AXIS_TEMPERATURE:
        return "temperature_K"
    if axis == AXIS_FIELD:
        return "magnetic_field_T"
    if axis == AXIS_INDEX:
        return "scan_index"
    if axis == AXIS_CUSTOM:
        name = re.sub(r"[^A-Za-z0-9_]+", "_", custom_field.strip()).strip("_")
        return name or "scan_axis"
    raise ValueError(f"Unknown scan axis {axis!r}")


def scan_axis_value(
    project: Project,
    axis: str,
    *,
    sequence_index: int | None = None,
    custom_field: str = "",
) -> float:
    if axis == AXIS_INDEX:
        if sequence_index is None:
            raise ValueError("Sequence index is unavailable")
        return float(sequence_index)
    if axis == AXIS_TEMPERATURE:
        value = _metadata_value(
            project, ("temperature", "sample temperature", "sample_temperature", "temp"))
        numeric, unit = _quantity(value)
        if unit in ("", "k", "kelvin"):
            return numeric
        if unit in ("mk", "millikelvin"):
            return numeric * 1e-3
        if unit in ("c", "°c", "celsius"):
            return numeric + 273.15
        raise ValueError(f"Unsupported temperature unit in {value!r}")
    if axis == AXIS_FIELD:
        value = _metadata_value(
            project, ("field", "magnetic field", "sample magnetic field", "sample_field"))
        numeric, unit = _quantity(value)
        factors = {
            "": 1.0,
            "t": 1.0,
            "tesla": 1.0,
            "mt": 1e-3,
            "millitesla": 1e-3,
            "ut": 1e-6,
            "gauss": 1e-4,
            "g": 1e-4,
            "kg": 0.1,
            "oe": 1e-4,
        }
        if unit not in factors:
            raise ValueError(f"Unsupported magnetic-field unit in {value!r}")
        return numeric * factors[unit]
    if axis == AXIS_CUSTOM:
        field = custom_field.strip()
        if not field:
            raise ValueError("A custom metadata field is required")
        numeric, _unit = _quantity(_metadata_value(project, (field,)))
        return numeric
    raise ValueError(f"Unknown scan axis {axis!r}")


def result_row(
    project: Project,
    axis: str,
    *,
    sequence_index: int | None = None,
    custom_field: str = "",
    project_path: Path | None = None,
) -> ResultRow:
    record = project.last_fit
    if record is not None and not record.valid:
        raise ValueError(record.message or "Project's saved fit is invalid")
    schema = parameter_schema(project)
    if record is None:
        values = {parameter.index: parameter.value for parameter in project.parameters}
        errors = {parameter.index: parameter.step for parameter in project.parameters}
    else:
        values = record.values
        errors = record.errors
    missing = [spec.index for spec in schema if spec.index not in values]
    missing_errors = [spec.index for spec in schema if spec.index not in errors]
    if missing or missing_errors:
        raise ValueError(
            f"Saved fit result is missing values/errors for parameters "
            f"{sorted(set(missing + missing_errors))}")
    return ResultRow(
        axis_value=scan_axis_value(
            project,
            axis,
            sequence_index=sequence_index,
            custom_field=custom_field,
        ),
        values=dict(values),
        errors=dict(errors),
        project_path=project_path,
    )


def collect_project_results(
    paths: Iterable[str | Path],
    axis: str,
    *,
    custom_field: str = "",
) -> tuple[tuple[ParameterSpec, ...], list[ResultRow]]:
    projects = []
    for sequence_index, source in enumerate(paths, 1):
        path = Path(source).expanduser().resolve()
        project = Project.load(path)
        projects.append((path, project, sequence_index))
    if not projects:
        raise ValueError("Select at least one PySRFit project")
    schema = parameter_schema(projects[0][1])
    rows = []
    for path, project, sequence_index in projects:
        candidate = parameter_schema(project)
        if candidate != schema:
            raise ValueError(
                f"Parameter schema in {path.name} does not match the first project")
        rows.append(result_row(
            project,
            axis,
            sequence_index=sequence_index,
            custom_field=custom_field,
            project_path=path,
        ))
    rows.sort(key=lambda row: row.axis_value)
    return schema, rows


def _column_name(spec: ParameterSpec) -> str:
    name = re.sub(r"[^A-Za-z0-9_]+", "_", spec.name).strip("_") or "parameter"
    return f"p{spec.index}_{name}"


def write_results_table(
    path: str | Path,
    axis: str,
    schema: tuple[ParameterSpec, ...],
    rows: Iterable[ResultRow],
    *,
    custom_field: str = "",
) -> Path:
    destination = Path(path).expanduser().resolve()
    headers = [axis_header(axis, custom_field)]
    for spec in schema:
        column = _column_name(spec)
        headers.extend((column, f"{column}_err"))
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(headers)
        for row in sorted(rows, key=lambda item: item.axis_value):
            values = [format(row.axis_value, ".15g")]
            for spec in schema:
                values.extend((
                    format(row.values[spec.index], ".15g"),
                    format(row.errors[spec.index], ".15g"),
                ))
            writer.writerow(values)
    return destination
