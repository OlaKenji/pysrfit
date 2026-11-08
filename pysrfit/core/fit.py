"""MINUIT2 single/global least-squares fitting and fit-range selection."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Tuple, Union

import numpy as np

from ..io.base import CountSpectrum, Spectrum
from ..io.asymmetry import asymmetry_from_components
from ..models import Project, RunConfig
from .theory import evaluate_theory


FitData = Union[Spectrum, CountSpectrum]
FitDataset = Tuple[RunConfig, FitData]


@dataclass
class FitResult:
    values: dict[int, float]
    errors: dict[int, float]
    chi2: float
    ndf: int
    valid: bool
    run_chi2: list[float] = field(default_factory=list)
    run_points: list[int] = field(default_factory=list)


def _parameter_reference(run: RunConfig, key: str) -> int:
    tokens = run.tokens(key)
    if not tokens:
        raise ValueError(f"Histogram fittype 0 requires RUN '{key}'")
    try:
        return int(float(tokens[0]))
    except ValueError as exc:
        raise NotImplementedError(f"RUN '{key}' functions are not implemented yet") from exc


def _mapping(run: RunConfig | None) -> list[int]:
    if run is None:
        return []
    try:
        return [int(float(value)) for value in run.tokens("map")]
    except ValueError as exc:
        raise ValueError("RUN map entries must be parameter numbers") from exc


def evaluate_data_model(
    project: Project,
    data: FitData,
    time_us,
    parameter_values: dict[int, float] | None = None,
    *,
    run: RunConfig | None = None,
) -> np.ndarray:
    """Evaluate one run with its own map and histogram settings."""
    values = project.parameter_values() if parameter_values is None else parameter_values
    time = np.asarray(time_us, dtype=float)
    selected_run = run or (project.runs[0] if project.runs else None)
    asymmetry = evaluate_theory(
        project.theory,
        time,
        values,
        project.functions_text,
        _mapping(selected_run),
    )
    if isinstance(data, Spectrum):
        return asymmetry
    if selected_run is None or selected_run.fittype != 0:
        raise ValueError("Count data requires a fittype 0 RUN block")
    norm_index = _parameter_reference(selected_run, "norm")
    if norm_index not in values:
        raise ValueError(f"RUN norm references missing parameter {norm_index}")
    lifetime_tokens = selected_run.tokens("lifetime")
    if lifetime_tokens:
        lifetime_index = _parameter_reference(selected_run, "lifetime")
        if lifetime_index not in values:
            raise ValueError(f"RUN lifetime references missing parameter {lifetime_index}")
        lifetime = values[lifetime_index]
    else:
        lifetime = 2.197019
    if lifetime <= 0:
        raise ValueError("Muon lifetime must be positive")
    background_tokens = selected_run.tokens("backgr.fit")
    if background_tokens:
        background_index = _parameter_reference(selected_run, "backgr.fit")
        if background_index not in values:
            raise ValueError(f"RUN backgr.fit references missing parameter {background_index}")
        background = values[background_index]
    else:
        background = float(data.metadata.get("background_per_packed_bin", 0.0))
    return values[norm_index] * np.exp(-time / lifetime) * (1.0 + asymmetry) + background


def _fit_range(project: Project, run: RunConfig) -> tuple[float, float]:
    tokens = run.tokens("fit")
    if len(tokens) >= 2:
        try:
            return float(tokens[0]), float(tokens[1])
        except ValueError as exc:
            raise ValueError("RUN fit range must contain two numbers") from exc
    # Plot limits are deliberately independent from fitting. With no explicit
    # per-run fit range, use every prepared point.
    return -np.inf, np.inf


def _usable_data(project: Project, run: RunConfig, data: FitData):
    tstart, tstop = _fit_range(project, run)
    if tstop <= tstart:
        raise ValueError(f"Invalid fit range for {run.label or run.source or 'run'}")
    good = (
        (data.time_us >= tstart)
        & (data.time_us <= tstop)
        & np.isfinite(data.values)
        & np.isfinite(data.error)
        & (data.error > 0)
    )
    indices = np.flatnonzero(good)
    return indices, data.time_us[good], data.values[good], data.error[good]


def _alpha_reference(run: RunConfig) -> int | None:
    tokens = run.tokens("alpha")
    if not tokens:
        return None
    try:
        return int(float(tokens[0]))
    except ValueError as exc:
        raise ValueError("RUN alpha must reference a parameter number") from exc


def fit_global(project: Project, datasets: Sequence[FitDataset]) -> FitResult:
    """Fit all datasets with one unique project parameter vector."""
    try:
        from iminuit import Minuit
    except ImportError as exc:
        raise RuntimeError("Fitting requires iminuit: pip install iminuit") from exc
    if not project.parameters:
        raise ValueError("No model parameters")
    if not datasets:
        raise ValueError("No fit-ready runs")

    pars = sorted(project.parameters, key=lambda parameter: parameter.index)
    names = [f"p{parameter.index}" for parameter in pars]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate parameter indices")
    free_n = sum(not parameter.fixed for parameter in pars)
    prepared = []
    total_points = 0
    parameter_by_index = {parameter.index: parameter for parameter in pars}
    for run, data in datasets:
        indices, t, y, error = _usable_data(project, run, data)
        if not len(t):
            raise ValueError(f"No usable points for {run.label or run.source or 'run'}")
        alpha_index = (
            _alpha_reference(run)
            if isinstance(data, Spectrum) else None
        )
        if alpha_index is not None:
            if alpha_index not in parameter_by_index:
                raise ValueError(f"RUN alpha references missing parameter {alpha_index}")
            if ("asymmetry_components" not in data.metadata
                    and not parameter_by_index[alpha_index].fixed):
                raise ValueError(
                    "Floating alpha needs raw detector components; reload the raw run")
        prepared.append((run, data, indices, t, y, error, alpha_index))
        total_points += len(t)
    if total_points <= free_n:
        raise ValueError("Too few usable data points for the number of free parameters")

    def contributions(values):
        parameter_values = {
            parameter.index: float(value)
            for parameter, value in zip(pars, values)
        }
        result = []
        for run, data, indices, time, observed, error, alpha_index in prepared:
            if alpha_index is not None and "asymmetry_components" in data.metadata:
                try:
                    all_observed, all_error = asymmetry_from_components(
                        data, parameter_values[alpha_index])
                except ValueError:
                    result.append(1e100)
                    continue
                observed = all_observed[indices]
                error = all_error[indices]
            model = evaluate_data_model(
                project, data, time, parameter_values, run=run)
            residual = (observed - model) / error
            result.append(float(np.dot(residual, residual)))
        return result

    def chi2(*values):
        return float(sum(contributions(values)))

    minimizer = Minuit(chi2, *[parameter.value for parameter in pars], name=names)
    minimizer.errordef = Minuit.LEAST_SQUARES
    for parameter, name in zip(pars, names):
        minimizer.fixed[name] = parameter.fixed
        if parameter.lower is not None or parameter.upper is not None:
            minimizer.limits[name] = (parameter.lower, parameter.upper)
        if parameter.step > 0:
            minimizer.errors[name] = parameter.step
    minimizer.migrad()
    if minimizer.valid:
        minimizer.hesse()
    fitted_values = [float(minimizer.values[name]) for name in names]
    run_chi2 = contributions(fitted_values)
    return FitResult(
        values={parameter.index: value for parameter, value in zip(pars, fitted_values)},
        errors={
            parameter.index: (
                0.0 if parameter.fixed else float(minimizer.errors[name]))
            for parameter, name in zip(pars, names)
        },
        chi2=float(sum(run_chi2)),
        ndf=total_points - free_n,
        valid=bool(minimizer.valid),
        run_chi2=run_chi2,
        run_points=[len(item[3]) for item in prepared],
    )
