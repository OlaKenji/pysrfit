"""Single-run sequential fitting built on persistent project fit records."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Callable, Iterable

from .core.fit import fit_global
from .models import FitRecord, Project
from .parameters import DEFAULT_SIGNIFICANT_DIGITS, format_parameter_text
from .results import (
    ParameterSpec,
    ResultRow,
    parameter_schema,
    result_row,
    write_results_table,
)
from .run_numbers import infer_run_number, path_for_run_number
from .session import AnalysisSession


@dataclass(frozen=True)
class SequentialTarget:
    run_number: int | None
    source_path: Path
    project_path: Path
    is_reference: bool = False


@dataclass(frozen=True)
class SequentialOutcome:
    target: SequentialTarget
    fit_valid: bool
    message: str = ""


@dataclass(frozen=True)
class SequentialSummary:
    result_path: Path
    outcomes: tuple[SequentialOutcome, ...]
    successful_rows: int
    cancelled: bool = False


_RUN_ITEM = re.compile(r"^(\d+)(?:-(\d+)(?::(\d+))?)?$")


def parse_run_numbers(text: str) -> list[int]:
    """Parse comma/space-separated numbers and inclusive ranges with optional step."""
    stripped = text.strip()
    if not stripped:
        return []
    result = []
    seen = set()
    for token in re.split(r"[,;\s]+", stripped):
        if not token:
            continue
        match = _RUN_ITEM.fullmatch(token)
        if match is None:
            raise ValueError(
                f"Invalid run-number item {token!r}; use e.g. 3092-3100,3105")
        start = int(match.group(1))
        stop = int(match.group(2)) if match.group(2) is not None else start
        step = int(match.group(3)) if match.group(3) is not None else 1
        if step <= 0:
            raise ValueError("Run-number range steps must be positive")
        direction = 1 if stop >= start else -1
        for number in range(start, stop + direction, direction * step):
            if number not in seen:
                result.append(number)
                seen.add(number)
    return result


def _reference_source(project: Project) -> Path:
    session = AnalysisSession(copy.deepcopy(project))
    sources = session.discover_project_data_sources()
    if len(sources) != 1 or sources[0] is None:
        raise ValueError("The reference project must have one configured data file")
    return sources[0].resolve()


def sequential_plan(project: Project, run_numbers: Iterable[int]) -> tuple[
        tuple[SequentialTarget, ...], Path]:
    if len(project.runs) != 1:
        raise ValueError("Sequential fitting requires exactly one run")
    if project.location is None:
        raise ValueError("Save the reference project before starting a sequential fit")
    reference_source = _reference_source(project)
    reference_number = infer_run_number(
        reference_source, project.runs[0].file_metadata)
    output_directory = project.location.parent
    targets = [SequentialTarget(
        reference_number,
        reference_source,
        output_directory / f"{reference_source.stem}.pysrfit",
        True,
    )]
    seen = {reference_number} if reference_number is not None else set()
    for number in run_numbers:
        if number in seen:
            continue
        source = path_for_run_number(
            reference_source, number, project.runs[0].file_metadata)
        targets.append(SequentialTarget(
            number,
            source,
            output_directory / f"{source.stem}.pysrfit",
        ))
        seen.add(number)
    result_path = output_directory / f"{project.location.stem}_results.csv"
    return tuple(targets), result_path


def sequential_existing_outputs(
    project: Project,
    run_numbers: Iterable[int],
) -> list[Path]:
    targets, result_path = sequential_plan(project, run_numbers)
    paths = [target.project_path for target in targets]
    paths.append(result_path)
    return [path for path in paths if path.exists()]


def _fit_record(result, message: str = "") -> FitRecord:
    return FitRecord(
        valid=bool(result.valid),
        chi2=result.chi2,
        ndf=result.ndf,
        values=result.values,
        errors=result.errors,
        run_chi2=result.run_chi2,
        run_points=result.run_points,
        message=message,
    )


def _format_fitted_parameters(project: Project) -> None:
    digits = int(project.display.get(
        "parameter_significant_digits", DEFAULT_SIGNIFICANT_DIGITS))
    project.parameter_text = format_parameter_text(
        project.parameters,
        project.parameter_text,
        significant_digits=digits,
    )


def run_sequential_fit(
    reference: Project,
    run_numbers: Iterable[int],
    axis: str,
    *,
    custom_field: str = "",
    chain_fit: bool = True,
    continue_on_error: bool = True,
    overwrite: bool = False,
    progress: Callable[[int, int, str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> SequentialSummary:
    """Fit a sequence cloned from a saved, single-run reference project."""
    targets, result_path = sequential_plan(reference, run_numbers)
    existing = [target.project_path for target in targets if target.project_path.exists()]
    if result_path.exists():
        existing.append(result_path)
    if existing and not overwrite:
        raise FileExistsError(
            "Sequential outputs already exist: "
            + ", ".join(path.name for path in existing))

    template = copy.deepcopy(reference)
    chain_values: dict[int, float] | None = None
    outcomes = []
    rows: list[ResultRow] = []
    schema: tuple[ParameterSpec, ...] | None = None
    was_cancelled = False

    for sequence_index, target in enumerate(targets, 1):
        if cancelled is not None and cancelled():
            was_cancelled = True
            break
        if progress is not None:
            progress(
                sequence_index - 1,
                len(targets),
                f"Preparing {target.source_path.name}",
            )
        project = copy.deepcopy(template)
        project.location = reference.location
        session = AnalysisSession(project)
        run = project.runs[0]
        reference_name = Path(run.source).stem if run.source else ""
        try:
            if not target.source_path.is_file():
                raise FileNotFoundError(f"Data file not found: {target.source_path}")
            session.attach_data_source(target.source_path)
            if not run.label or run.label == reference_name:
                run.label = target.source_path.stem
            load = session.ensure_data_loaded(force=True)
            if load.processing_error:
                raise ValueError(load.processing_error)
            if chain_fit and chain_values:
                for parameter in project.parameters:
                    if parameter.index in chain_values:
                        parameter.value = chain_values[parameter.index]
                session.refresh_parameter_dependent_data()
            prepared = session.prepared_runs()
            if len(prepared) != 1:
                raise ValueError("The run did not produce one fit-ready dataset")
            _index, configured_run, data = prepared[0]
            if progress is not None:
                progress(
                    sequence_index - 1,
                    len(targets),
                    f"Fitting {target.source_path.name}",
                )
            result = fit_global(project, [(configured_run, data)])
            project.last_fit = _fit_record(
                result,
                "" if result.valid else "MINUIT did not converge to a valid minimum",
            )
            if result.valid:
                for parameter in project.parameters:
                    parameter.value = result.values[parameter.index]
                    if not parameter.fixed:
                        parameter.step = result.errors[parameter.index]
                _format_fitted_parameters(project)
                if chain_fit:
                    chain_values = dict(result.values)
            project.save(target.project_path)
            if not result.valid:
                message = project.last_fit.message
                outcomes.append(SequentialOutcome(target, False, message))
                if not continue_on_error:
                    break
                continue
            candidate_schema = parameter_schema(project)
            if schema is None:
                schema = candidate_schema
            elif candidate_schema != schema:
                raise ValueError("The fitted parameter schema changed between runs")
            try:
                rows.append(result_row(
                    project,
                    axis,
                    sequence_index=sequence_index,
                    custom_field=custom_field,
                    project_path=target.project_path,
                ))
                outcomes.append(SequentialOutcome(target, True))
            except ValueError as axis_error:
                outcomes.append(SequentialOutcome(
                    target,
                    True,
                    f"Fit saved, but omitted from table: {axis_error}",
                ))
                if not continue_on_error:
                    break
        except Exception as exc:
            project.runs[0].source = str(target.source_path)
            project.data_override = str(target.source_path)
            project.last_fit = FitRecord(valid=False, message=str(exc))
            project.save(target.project_path)
            outcomes.append(SequentialOutcome(target, False, str(exc)))
            if not continue_on_error:
                break

    if schema is None:
        schema = parameter_schema(template)
    write_results_table(
        result_path,
        axis,
        schema,
        rows,
        custom_field=custom_field,
    )
    if progress is not None:
        progress(len(outcomes), len(targets), "Sequential fit complete")
    return SequentialSummary(
        result_path=result_path,
        outcomes=tuple(outcomes),
        successful_rows=len(rows),
        cancelled=was_cancelled,
    )
