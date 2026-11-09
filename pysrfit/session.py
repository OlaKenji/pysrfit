"""Stateful multi-run analysis session, independent of Qt widgets."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Union

import numpy as np

from .io.asymmetry import asymmetry_from_components, prepare_run_data
from .io.base import CountSpectrum, HistogramRun, Spectrum
from .io.loaders import load_data
from .models import DEFAULT_PACKING, Parameter, Project, RunConfig, demo_project


PreparedData = Union[Spectrum, CountSpectrum]
DataLoaderFunction = Callable[[Union[str, Path]], Union[Spectrum, HistogramRun]]

_RUNTIME_METADATA_KEYS = {
    "source", "reader", "derived", "fit_type", "alpha", "packing",
    "forward", "backward", "t0", "background", "asymmetry_components",
    "background_per_packed_bin", "facility", "instrument", "format",
    "suggested_run_configuration",
}


def _metadata_value(value):
    """Convert small metadata values to JSON-compatible Python objects."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)) and len(value) <= 100:
        return [_metadata_value(item) for item in value]
    if isinstance(value, dict) and len(value) <= 100:
        return {str(key): _metadata_value(item) for key, item in value.items()}
    return None


def persistent_file_metadata(metadata: dict) -> dict:
    """Keep file-header metadata while excluding derived/runtime arrays."""
    result = {}
    for key, value in metadata.items():
        if key in _RUNTIME_METADATA_KEYS:
            continue
        converted = _metadata_value(value)
        if converted is not None:
            result[str(key)] = converted
    return result


@dataclass(frozen=True)
class LoadOutcome:
    loaded: bool
    reloaded: bool = False
    processing_error: str = ""
    added_parameters: tuple[int, ...] = ()


@dataclass
class RunDataState:
    """Runtime arrays and cache metadata for one persistent RUN block."""

    source_path: Path | None = None
    data: PreparedData | None = None
    raw_data: HistogramRun | None = None
    source_signature: tuple[str, int, int] | None = None
    loading: bool = False

    def clear_arrays(self) -> None:
        self.data = None
        self.raw_data = None
        self.source_signature = None


class AnalysisSession:
    """Own a project and independent runtime caches for all of its runs."""

    DATA_SUFFIXES = (".bin", ".BIN", ".mdu", ".nxs", ".h5", ".root", ".dat")

    def __init__(
        self,
        project: Project | None = None,
        *,
        loader: DataLoaderFunction = load_data,
    ):
        self.project = project or demo_project()
        self.selected_run_index = 0
        self._states = [RunDataState() for _ in self.project.runs] or [RunDataState()]
        self._loader = loader

    def _ensure_state_count(self) -> None:
        wanted = max(1, len(self.project.runs))
        while len(self._states) < wanted:
            self._states.append(RunDataState())
        if len(self._states) > wanted:
            del self._states[wanted:]
        self.selected_run_index = min(self.selected_run_index, wanted - 1)

    def state(self, index: int | None = None) -> RunDataState:
        self._ensure_state_count()
        selected = self.selected_run_index if index is None else index
        if not 0 <= selected < len(self._states):
            raise IndexError(f"Run index {selected} is out of range")
        return self._states[selected]

    @property
    def data(self) -> PreparedData | None:
        return self.state().data

    @data.setter
    def data(self, value: PreparedData | None) -> None:
        self.state().data = value

    @property
    def raw_data(self) -> HistogramRun | None:
        return self.state().raw_data

    @raw_data.setter
    def raw_data(self, value: HistogramRun | None) -> None:
        self.state().raw_data = value

    @property
    def source_path(self) -> Path | None:
        return self.state().source_path

    @source_path.setter
    def source_path(self, value: Path | None) -> None:
        self.state().source_path = value

    @property
    def current_run(self) -> RunConfig | None:
        if not self.project.runs:
            return None
        return self.project.runs[self.selected_run_index]

    @property
    def run_count(self) -> int:
        return len(self.project.runs)

    def select_run(self, index: int) -> None:
        self._ensure_state_count()
        if not self.project.runs:
            self.selected_run_index = 0
            return
        if not 0 <= index < len(self.project.runs):
            raise IndexError(f"Run index {index} is out of range")
        self.selected_run_index = index

    def replace_project(self, project: Project) -> None:
        self.project = project
        self.selected_run_index = 0
        self._states = [RunDataState() for _ in project.runs] or [RunDataState()]

    def clear_runtime_data(self) -> None:
        self._states = [RunDataState() for _ in self.project.runs] or [RunDataState()]
        self.selected_run_index = min(self.selected_run_index, len(self._states) - 1)

    @staticmethod
    def source_signature(path: Path) -> tuple[str, int, int]:
        stat = path.stat()
        return str(path), stat.st_mtime_ns, stat.st_size

    def attach_data_source(
        self,
        path: str | Path,
        *,
        persist: bool = True,
        new_run: bool = False,
    ) -> Path:
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Data file not found: {source}")

        if new_run and self.project.runs:
            self.project.runs.append(RunConfig(source=str(source), label=source.stem))
            self._states.append(RunDataState())
            self.selected_run_index = len(self.project.runs) - 1
        elif not self.project.runs:
            self.project.runs.append(RunConfig(source=str(source), label=source.stem))
            self._ensure_state_count()
            self.selected_run_index = 0

        run = self.project.runs[self.selected_run_index]
        run.source = str(source)
        if not run.label:
            run.label = source.stem
        state = self.state()
        state.source_path = source
        state.clear_arrays()
        if persist and self.selected_run_index == 0:
            self.project.data_override = str(source)
        return source

    def remove_run(self, index: int | None = None) -> None:
        if not self.project.runs:
            return
        selected = self.selected_run_index if index is None else index
        if not 0 <= selected < len(self.project.runs):
            raise IndexError(f"Run index {selected} is out of range")
        self.project.runs.pop(selected)
        self._states.pop(selected)
        if selected == 0:
            self.project.data_override = ""
            self.project.spectrum_path = ""
        if not self._states:
            self._states = [RunDataState()]
        self.selected_run_index = min(selected, max(0, len(self.project.runs) - 1))

    def _resolve_run_source(self, index: int) -> Path | None:
        run = self.project.runs[index]
        reference = run.source
        if index == 0 and self.project.data_override:
            reference = self.project.data_override
        if not reference and index == 0 and self.project.spectrum_path:
            reference = self.project.spectrum_path
        if not reference:
            return None
        base = self.project.resolve(reference)
        if base.is_file():
            return base
        for suffix in self.DATA_SUFFIXES:
            candidate = Path(str(base) + suffix)
            if candidate.is_file():
                return candidate
        return base

    def discover_project_data_sources(self) -> list[Path | None]:
        """Resolve configured sources without reading histogram arrays."""
        if not self.project.runs and (self.project.data_override or self.project.spectrum_path):
            reference = self.project.data_override or self.project.spectrum_path
            candidate = self.project.resolve(reference)
            self.project.runs.append(RunConfig(source=str(candidate), label=candidate.stem))
            self._ensure_state_count()
        self._ensure_state_count()
        sources = []
        for index in range(len(self.project.runs)):
            candidate = self._resolve_run_source(index)
            self._states[index].source_path = candidate
            sources.append(candidate)
        return sources

    def discover_project_data_source(self) -> Path | None:
        sources = self.discover_project_data_sources()
        return sources[self.selected_run_index] if sources else None

    def ensure_data_loaded(
        self,
        *,
        force: bool = False,
        index: int | None = None,
    ) -> LoadOutcome:
        selected = self.selected_run_index if index is None else index
        state = self.state(selected)
        if state.source_path is None and selected < len(self.project.runs):
            state.source_path = self._resolve_run_source(selected)
        if state.loading or state.source_path is None:
            return LoadOutcome(state.data is not None or state.raw_data is not None)
        if not state.source_path.is_file():
            raise FileNotFoundError(f"Data file not found: {state.source_path}")
        signature = self.source_signature(state.source_path)
        if (not force and signature == state.source_signature
                and (state.data is not None or state.raw_data is not None)):
            return LoadOutcome(False)
        reloaded = state.source_signature is not None
        state.loading = True
        try:
            loaded = self._loader(state.source_path)
            processing_error, added_parameters = self._accept_loaded_data(selected, loaded)
            state.source_signature = signature
        finally:
            state.loading = False
        return LoadOutcome(True, reloaded, processing_error, added_parameters)

    def ensure_all_data_loaded(self, *, force: bool = False) -> list[tuple[int, LoadOutcome]]:
        outcomes = []
        for index, _run in enumerate(self.project.runs):
            outcomes.append((index, self.ensure_data_loaded(force=force, index=index)))
        return outcomes

    def _accept_loaded_data(
        self,
        index: int,
        loaded: Spectrum | HistogramRun,
    ) -> tuple[str, tuple[int, ...]]:
        state = self.state(index)
        run = self.project.runs[index]
        if isinstance(loaded, Spectrum):
            state.data, state.raw_data = loaded, None
            run.file_metadata = persistent_file_metadata(loaded.metadata)
            if not run.format_hint:
                run.format_hint = str(loaded.metadata.get("reader", "spectrum"))
            return "", ()

        state.raw_data, state.data = loaded, None
        if not run.tokens("forward") and not run.tokens("backward"):
            self.project.runs[index] = self.default_run_for_raw_data(
                state.source_path or run.source, index=index)
            run = self.project.runs[index]
        else:
            run.source = str(state.source_path or run.source)
            run.instrument = run.instrument or str(loaded.metadata.get("instrument", ""))
            run.facility = run.facility or str(loaded.metadata.get("facility", ""))
            run.format_hint = run.format_hint or str(
                loaded.metadata.get("format", loaded.metadata.get("reader", "")))
            run.file_metadata = persistent_file_metadata(loaded.metadata)
        try:
            added_parameters = self.process_raw_data(index=index)
        except Exception as exc:
            state.data = None
            return str(exc), ()
        return "", added_parameters

    def default_run_for_raw_data(
        self,
        path: str | Path,
        *,
        index: int | None = None,
    ) -> RunConfig:
        selected = self.selected_run_index if index is None else index
        data = self.state(selected).raw_data
        if data is None:
            raise ValueError("No raw histogram data loaded")
        suggestion = data.metadata.get("suggested_run_configuration", {})
        if not isinstance(suggestion, dict):
            suggestion = {}
        fittype = int(suggestion.get("fittype", 2))
        if fittype not in (0, 2):
            fittype = 2

        def suggested_detectors(key: str) -> list:
            references = suggestion.get(key, [])
            if not isinstance(references, (list, tuple)):
                return []
            return [data.detectors[data.detector_index(reference)]
                    for reference in references]

        forward = suggested_detectors("forward")
        backward = suggested_detectors("backward") if fittype == 2 else []
        if not suggestion:
            named_forward = [
                detector for detector in data.detectors
                if detector.name.strip().casefold() in {"forw", "forward"}
            ]
            named_backward = [
                detector for detector in data.detectors
                if detector.name.strip().casefold() in {"back", "backward"}
            ]
            if len(named_forward) == len(named_backward) == 1:
                forward, backward = named_forward, named_backward
        if fittype == 2 and len(forward) != len(backward):
            raise ValueError(
                "Suggested asymmetry detector groups must have equal sizes")
        source = Path(path).expanduser().resolve()
        old = self.project.runs[selected] if selected < len(self.project.runs) else None
        run = RunConfig(
            source=str(source),
            label=(old.label if old and old.label else source.stem),
            plot_enabled=old.plot_enabled if old else True,
            instrument=str(data.metadata.get("instrument", "")),
            facility=str(data.metadata.get("facility", "")),
            format_hint=str(data.metadata.get("format", data.metadata.get("reader", ""))),
            fittype=fittype,
            file_metadata=persistent_file_metadata(data.metadata),
            user_metadata=dict(old.user_metadata) if old else {},
        )

        def detector_tokens(detectors) -> list[str]:
            numbers = [detector.number for detector in detectors]
            tokens: list[str] = []
            start = stop = None
            for number in numbers + [None]:
                if start is None:
                    start = stop = number
                elif number is not None and number == stop + 1:
                    stop = number
                else:
                    tokens.append(str(start) if start == stop else f"{start}-{stop}")
                    start = stop = number
            return tokens

        if forward:
            run.options["forward"] = detector_tokens(forward)
        if backward:
            run.options["backward"] = detector_tokens(backward)
        all_t0_available = bool(forward) and all(
            detector.t0_bin is not None for detector in forward + backward)
        # Large NeXus detector banks use the per-detector metadata directly.
        # Writing hundreds of identical t0 values into the editable RUN field
        # makes the project needlessly unwieldy.
        if all_t0_available and len(forward) + len(backward) <= 64:
            if backward:
                run.options["t0"] = [
                    str(detector.t0_bin)
                    for pair in zip(forward, backward)
                    for detector in pair
                ]
            else:
                run.options["t0"] = [
                    str(detector.t0_bin) for detector in forward]
        fixed_background = suggestion.get("backgr.fix")
        if (backward and isinstance(fixed_background, (list, tuple))
                and len(fixed_background) >= 2):
            run.options["backgr.fix"] = [
                str(fixed_background[0]), str(fixed_background[1])]
        elif (backward and all(
                key in detector.metadata
                for detector in (forward[0], backward[0])
                for key in ("background_start", "background_end"))):
            run.options["background"] = [
                str(forward[0].metadata["background_start"]),
                str(forward[0].metadata["background_end"]),
                str(backward[0].metadata["background_start"]),
                str(backward[0].metadata["background_end"]),
            ]
        elif backward and all_t0_available:
            run.options["background"] = [
                "0", str(max(0, int(forward[0].t0_bin) - 100)),
                "0", str(max(0, int(backward[0].t0_bin) - 100)),
            ]
        if backward and None not in (
            forward[0].first_good_bin, forward[0].last_good_bin,
            backward[0].first_good_bin, backward[0].last_good_bin,
        ):
            forward_stop = forward[0].last_good_bin
            backward_stop = backward[0].last_good_bin
            suggested_stop_us = suggestion.get("data_stop_us")
            if suggested_stop_us is not None and all_t0_available:
                span = max(float(suggested_stop_us), data.channel_width_us)
                forward_stop = min(
                    forward_stop,
                    int(forward[0].t0_bin + span / data.channel_width_us),
                )
                backward_stop = min(
                    backward_stop,
                    int(backward[0].t0_bin + span / data.channel_width_us),
                )
            run.options["data"] = [
                str(forward[0].first_good_bin), str(forward_stop),
                str(backward[0].first_good_bin), str(backward_stop),
            ]
        elif forward and not backward and None not in (
            forward[0].first_good_bin, forward[0].last_good_bin,
        ):
            forward_stop = forward[0].last_good_bin
            suggested_stop_us = suggestion.get("data_stop_us")
            if suggested_stop_us is not None and all_t0_available:
                span = max(float(suggested_stop_us), data.channel_width_us)
                forward_stop = min(
                    forward_stop,
                    int(forward[0].t0_bin + span / data.channel_width_us),
                )
            run.options["data"] = [
                str(forward[0].first_good_bin),
                str(forward_stop),
            ]
        packing = int(suggestion.get("packing", DEFAULT_PACKING))
        run.options["packing"] = [str(max(1, packing))]
        if old:
            for key in ("map", "alpha", "fit"):
                if old.tokens(key):
                    run.options[key] = list(old.tokens(key))
        return run

    def ensure_alpha_parameter(self, index: int) -> int | None:
        """Ensure a type-2 run references a positive, floating alpha parameter."""
        run = self.project.runs[index]
        if run.fittype != 2:
            return None
        tokens = run.tokens("alpha")
        if tokens:
            try:
                numeric_index = float(tokens[0])
            except ValueError as exc:
                raise ValueError(
                    "alpha must reference an existing fit parameter") from exc
            if (not np.isfinite(numeric_index) or not numeric_index.is_integer()
                    or numeric_index < 1):
                raise ValueError(
                    "alpha must reference a positive integer fit parameter number")
            parameter_index = int(numeric_index)
            if any(parameter.index == parameter_index
                   for parameter in self.project.parameters):
                return None
            # Repair a valid numeric reference left dangling by an interrupted
            # or older detector-processing attempt.
            base_name = f"alpha_run{index + 1}"
            existing_names = {parameter.name for parameter in self.project.parameters}
            name = (
                base_name if base_name not in existing_names
                else f"{base_name}_parameter{parameter_index}"
            )
            self.project.parameters.append(Parameter(
                parameter_index, name, 1.0, 0.01, 0.01, 10.0))
            return parameter_index
        name = f"alpha_run{index + 1}"
        existing = next(
            (parameter for parameter in self.project.parameters if parameter.name == name),
            None,
        )
        if existing is None:
            parameter_index = max(
                (parameter.index for parameter in self.project.parameters), default=0) + 1
            self.project.parameters.append(Parameter(
                parameter_index, name, 1.0, 0.01, 0.01, 10.0))
            added = parameter_index
        else:
            parameter_index = existing.index
            added = None
        run.options["alpha"] = [str(parameter_index)]
        return added

    def alpha_parameter(self, index: int) -> Parameter | None:
        """Return the parameter referenced by a type-2 RUN's alpha setting."""
        run = self.project.runs[index]
        tokens = run.tokens("alpha")
        if run.fittype != 2 or not tokens:
            return None
        try:
            numeric_index = float(tokens[0])
        except ValueError as exc:
            raise ValueError("alpha must reference an existing fit parameter") from exc
        if not np.isfinite(numeric_index) or not numeric_index.is_integer():
            raise ValueError("alpha must reference an integer fit parameter number")
        parameter_index = int(numeric_index)
        parameter = next(
            (item for item in self.project.parameters if item.index == parameter_index),
            None,
        )
        if parameter is None:
            raise ValueError(
                f"alpha references missing fit parameter {parameter_index}")
        return parameter

    def refresh_parameter_dependent_data(self) -> int:
        """Refresh cached spectra whose measured values depend on parameters.

        Alpha changes only require recombining the already prepared forward and
        backward detector components.  This is both faster and less error-prone
        than rebuilding detector groups, backgrounds, t0 alignment and packing.
        """
        refreshed = 0
        for index, run in enumerate(self.project.runs):
            data = self.state(index).data
            if run.fittype != 2 or not isinstance(data, Spectrum):
                continue
            if "asymmetry_components" not in data.metadata:
                continue
            parameter = self.alpha_parameter(index)
            if parameter is None:
                continue
            asymmetry, error = asymmetry_from_components(data, parameter.value)
            data.asymmetry = asymmetry
            data.error = error
            data.metadata["alpha"] = parameter.value
            refreshed += 1
        return refreshed

    def ensure_histogram_parameters(self, index: int) -> list[int]:
        """Ensure a type-0 run has normalization and fitted-background parameters."""
        run = self.project.runs[index]
        data = self.state(index).data
        if run.fittype != 0 or not isinstance(data, CountSpectrum):
            return []

        background = float(data.metadata.get("background_per_packed_bin", 0.0))
        estimates = {
            "norm": (
                f"N0_run{index + 1}",
                max(float(np.max(data.counts)) - background, 1.0),
                0.0,
            ),
            "backgr.fit": (
                f"background_run{index + 1}",
                max(background, 0.0),
                0.0,
            ),
        }
        added = []
        for option, (name, estimate, lower) in estimates.items():
            if run.tokens(option):
                continue
            existing = next(
                (parameter for parameter in self.project.parameters
                 if parameter.name == name),
                None,
            )
            if existing is None:
                parameter_index = max(
                    (parameter.index for parameter in self.project.parameters),
                    default=0,
                ) + 1
                self.project.parameters.append(Parameter(
                    parameter_index,
                    name,
                    estimate,
                    max(np.sqrt(max(estimate, 1.0)), 1.0),
                    lower,
                    None,
                ))
                added.append(parameter_index)
            else:
                parameter_index = existing.index
            run.options[option] = [str(parameter_index)]
        return added

    def process_raw_data(self, *, index: int | None = None) -> tuple[int, ...]:
        selected = self.selected_run_index if index is None else index
        state = self.state(selected)
        if state.raw_data is None or selected >= len(self.project.runs):
            raise ValueError("No raw histogram data and RUN configuration are available")
        run = self.project.runs[selected]
        if not run.tokens("forward"):
            raise ValueError(
                "Select a fit type and detector(s) for this unrecognized layout")
        if run.fittype == 2 and not run.tokens("backward"):
            raise ValueError(
                "Forward/backward asymmetry requires a backward detector group")
        parameters_before = list(self.project.parameters)
        options_before = copy.deepcopy(run.options)
        data_before = state.data
        try:
            added_parameters = []
            added_alpha = self.ensure_alpha_parameter(selected)
            if added_alpha is not None:
                added_parameters.append(added_alpha)
            alpha = 1.0
            parameter = self.alpha_parameter(selected)
            if parameter is not None:
                alpha = parameter.value
            state.data = prepare_run_data(state.raw_data, run, alpha=alpha)
            added_parameters.extend(self.ensure_histogram_parameters(selected))
            return tuple(added_parameters)
        except Exception:
            # Parameter creation is part of successful detector preparation.
            # Do not leave invisible parameters or dangling RUN references
            # behind when t0/range validation fails.
            self.project.parameters[:] = parameters_before
            run.options = options_before
            state.data = data_before
            raise

    def prepare_view_data(
        self,
        packing: int,
        *,
        index: int | None = None,
    ) -> PreparedData:
        """Prepare display-only data without changing the fitted spectrum.

        Raw histogram runs can be prepared again at any requested packing.
        Already processed spectrum files have no underlying histogram bins, so
        their stored packing is retained.
        """
        selected = self.selected_run_index if index is None else index
        state = self.state(selected)
        if state.data is None or selected >= len(self.project.runs):
            raise ValueError("No fit-ready data are available")
        view_packing = int(packing)
        if view_packing < 1:
            raise ValueError("View packing must be at least one")
        fit_packing = int(state.data.metadata.get("packing", 1))
        if state.raw_data is None or view_packing == fit_packing:
            return state.data

        run = copy.deepcopy(self.project.runs[selected])
        run.options["packing"] = [str(view_packing)]
        alpha = 1.0
        parameter = self.alpha_parameter(selected)
        if parameter is not None:
            alpha = parameter.value
        return prepare_run_data(state.raw_data, run, alpha=alpha)

    def prepared_runs(
        self,
        *,
        plotted_only: bool = False,
    ) -> list[tuple[int, RunConfig, PreparedData]]:
        self._ensure_state_count()
        result = []
        for index, run in enumerate(self.project.runs):
            if plotted_only and not run.plot_enabled:
                continue
            data = self._states[index].data
            if data is not None:
                result.append((index, run, data))
        return result
