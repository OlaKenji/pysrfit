"""MINUIT2 least-squares fitting and fit-range selection for mysr."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ..models import Project
from ..io.base import CountSpectrum, Spectrum
from .theory import evaluate_theory


@dataclass
class FitResult:
    values: dict[int, float]
    errors: dict[int, float]
    chi2: float
    ndf: int
    valid: bool


def _parameter_reference(run, key: str) -> int:
    tokens = run.tokens(key)
    if not tokens:
        raise ValueError(f"Histogram fittype 0 requires RUN '{key}'")
    try:
        return int(float(tokens[0]))
    except ValueError as exc:
        raise NotImplementedError(f"RUN '{key}' functions are not implemented yet") from exc


def evaluate_data_model(
    project: Project,
    data: Spectrum | CountSpectrum,
    time_us,
    parameter_values: dict[int, float] | None = None,
) -> np.ndarray:
    """Evaluate either asymmetry A(t) or the musrfit type-0 count model."""
    values = project.parameter_values() if parameter_values is None else parameter_values
    time = np.asarray(time_us, dtype=float)
    asymmetry = evaluate_theory(
        project.theory, time, values, project.functions_text, _mapping(project))
    if isinstance(data, Spectrum):
        return asymmetry
    if not project.runs or project.runs[0].fittype != 0:
        raise ValueError("Count data requires a fittype 0 RUN block")
    run = project.runs[0]
    norm_index = _parameter_reference(run, "norm")
    if norm_index not in values:
        raise ValueError(f"RUN norm references missing parameter {norm_index}")
    lifetime_tokens = run.tokens("lifetime")
    if lifetime_tokens:
        lifetime_index = _parameter_reference(run, "lifetime")
        if lifetime_index not in values:
            raise ValueError(f"RUN lifetime references missing parameter {lifetime_index}")
        lifetime = values[lifetime_index]
    else:
        lifetime = 2.197019
    if lifetime <= 0:
        raise ValueError("Muon lifetime must be positive")
    background_tokens = run.tokens("backgr.fit")
    if background_tokens:
        background_index = _parameter_reference(run, "backgr.fit")
        if background_index not in values:
            raise ValueError(f"RUN backgr.fit references missing parameter {background_index}")
        background = values[background_index]
    else:
        background = float(data.metadata.get("background_per_packed_bin", 0.0))
    return values[norm_index] * np.exp(-time / lifetime) * (1.0 + asymmetry) + background


def fit_project(project: Project, data: Spectrum | CountSpectrum) -> FitResult:
    try:
        from iminuit import Minuit
    except ImportError as exc:
        raise RuntimeError("Fitting requires iminuit: pip install iminuit") from exc
    if not project.parameters:
        raise ValueError("No model parameters")
    tstart = float(project.display.get("t_start", 0))
    tstop = float(project.display.get("t_stop", 1e12))
    good = ((data.time_us >= tstart) & (data.time_us <= tstop) &
            np.isfinite(data.values) & np.isfinite(data.error) & (data.error > 0))
    t, y, e = data.time_us[good], data.values[good], data.error[good]
    pars = sorted(project.parameters, key=lambda x: x.index)
    free_n = sum(not p.fixed for p in pars)
    if len(t) <= free_n:
        raise ValueError("Too few usable data points for the number of free parameters")
    names = [f"p{p.index}" for p in pars]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate parameter indices")

    # Explicitly refuse varying alpha for precomputed detector asymmetries.
    # For floating alpha, data and errors must be recalculated within the likelihood.
    alpha_keys = {int(float(x)) for run in project.runs for x in run.tokens("alpha") if x.isdigit()}
    if data.metadata.get("derived") == "asymmetry" and any(not p.fixed and p.index in alpha_keys for p in pars):
        raise NotImplementedError("Floating alpha requires dynamic asymmetry reconstruction; fix alpha for now")

    def chi2(*v):
        pvals = {p.index: float(x) for p, x in zip(pars, v)}
        model = evaluate_data_model(project, data, t, pvals)
        resid = (y - model) / e
        return float(np.dot(resid, resid))

    m = Minuit(chi2, *[p.value for p in pars], name=names)
    m.errordef = Minuit.LEAST_SQUARES
    for p, name in zip(pars, names):
        m.fixed[name] = p.fixed
        if p.lower is not None or p.upper is not None:
            m.limits[name] = (p.lower, p.upper)
        if p.step > 0:
            m.errors[name] = p.step
    m.migrad()
    if m.valid:
        m.hesse()
    return FitResult(values={p.index: float(m.values[n]) for p, n in zip(pars, names)},
                     errors={p.index: float(m.errors[n]) for p, n in zip(pars, names)},
                     chi2=float(m.fval), ndf=len(t)-free_n, valid=bool(m.valid))


def _mapping(project: Project) -> list[int]:
    if not project.runs:
        return []
    return [int(float(x)) for x in project.runs[0].tokens("map")]
