"""Stateful analysis session, independent of Qt widgets."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Union

import numpy as np

from .io.asymmetry import prepare_run_data
from .io.base import CountSpectrum, HistogramRun, Spectrum
from .io.loaders import load_data
from .models import DEFAULT_PACKING, Parameter, Project, RunConfig, demo_project


PreparedData = Union[Spectrum, CountSpectrum]
DataLoaderFunction = Callable[[Union[str, Path]], Union[Spectrum, HistogramRun]]


@dataclass(frozen=True)
class LoadOutcome:
    loaded: bool
    reloaded: bool = False
    processing_error: str = ""
    normalization_parameter: int | None = None


class AnalysisSession:
    """Own one analysis project and its non-persistent runtime data cache."""

    DATA_SUFFIXES = (".bin", ".BIN", ".mdu", ".nxs", ".h5", ".root", ".dat")

    def __init__(
        self,
        project: Project | None = None,
        *,
        loader: DataLoaderFunction = load_data,
    ):
        self.project = project or demo_project()
        self.data: PreparedData | None = None
        self.raw_data: HistogramRun | None = None
        self.source_path: Path | None = None
        self._source_signature: tuple[str, int, int] | None = None
        self._loading = False
        self._loader = loader

    def replace_project(self, project: Project) -> None:
        self.project = project
        self.clear_runtime_data()

    def clear_runtime_data(self) -> None:
        self.data = None
        self.raw_data = None
        self.source_path = None
        self._source_signature = None

    @staticmethod
    def source_signature(path: Path) -> tuple[str, int, int]:
        stat = path.stat()
        return str(path), stat.st_mtime_ns, stat.st_size

    def attach_data_source(self, path: str | Path, *, persist: bool = True) -> Path:
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Data file not found: {source}")
        self.source_path = source
        self.data = None
        self.raw_data = None
        self._source_signature = None
        if persist:
            self.project.data_override = str(source)
        return source

    def discover_project_data_source(self) -> Path | None:
        """Resolve the configured project source without reading its arrays."""
        if self.project.data_override:
            candidate = self.project.resolve(self.project.data_override)
            self.source_path = candidate
            return candidate
        if self.project.spectrum_path:
            candidate = self.project.resolve(self.project.spectrum_path)
            self.source_path = candidate
            return candidate
        if self.project.runs and self.project.runs[0].source:
            base = self.project.resolve(self.project.runs[0].source)
            for candidate in (base, *(Path(str(base) + suffix) for suffix in self.DATA_SUFFIXES)):
                if candidate.is_file():
                    self.source_path = candidate
                    return candidate
        return None

    def ensure_data_loaded(self, *, force: bool = False) -> LoadOutcome:
        if self._loading or self.source_path is None:
            return LoadOutcome(self.data is not None or self.raw_data is not None)
        if not self.source_path.is_file():
            raise FileNotFoundError(f"Data file not found: {self.source_path}")
        signature = self.source_signature(self.source_path)
        if (not force and signature == self._source_signature
                and (self.data is not None or self.raw_data is not None)):
            return LoadOutcome(False)
        reloaded = self._source_signature is not None
        self._loading = True
        try:
            loaded = self._loader(self.source_path)
            processing_error, normalization = self._accept_loaded_data(loaded)
            self._source_signature = signature
        finally:
            self._loading = False
        return LoadOutcome(True, reloaded, processing_error, normalization)

    def _accept_loaded_data(self, loaded: Spectrum | HistogramRun) -> tuple[str, int | None]:
        if isinstance(loaded, Spectrum):
            self.data, self.raw_data = loaded, None
            return "", None
        self.raw_data, self.data = loaded, None
        if not self.project.runs:
            if self.source_path is None:
                raise ValueError("Raw histogram data has no source path")
            self.project.runs.append(self.default_run_for_raw_data(self.source_path))
        try:
            normalization = self.process_raw_data()
        except Exception as exc:
            self.data = None
            return str(exc), None
        return "", normalization

    def default_run_for_raw_data(self, path: str | Path) -> RunConfig:
        data = self.raw_data
        if data is None:
            raise ValueError("No raw histogram data loaded")
        forward = next(
            (detector for detector in data.detectors if detector.name.casefold().startswith("forw")),
            data.detectors[0],
        )
        backward = next(
            (detector for detector in data.detectors if detector.name.casefold().startswith("back")),
            data.detectors[min(1, len(data.detectors) - 1)],
        )
        run = RunConfig(
            source=str(Path(path).expanduser().resolve()),
            instrument=str(data.metadata.get("instrument", "")),
            facility=str(data.metadata.get("facility", "")),
            format_hint=str(data.metadata.get("format", data.metadata.get("reader", ""))),
            fittype=2,
        )
        run.options["forward"] = [str(forward.number)]
        run.options["backward"] = [str(backward.number)]
        if forward.t0_bin is not None and backward.t0_bin is not None:
            run.options["t0"] = [str(forward.t0_bin), str(backward.t0_bin)]
            run.options["background"] = [
                "0", str(max(0, int(forward.t0_bin) - 100)),
                "0", str(max(0, int(backward.t0_bin) - 100)),
            ]
        if None not in (
            forward.first_good_bin, forward.last_good_bin,
            backward.first_good_bin, backward.last_good_bin,
        ):
            run.options["data"] = [
                str(forward.first_good_bin), str(forward.last_good_bin),
                str(backward.first_good_bin), str(backward.last_good_bin),
            ]
        run.options["packing"] = [str(DEFAULT_PACKING)]
        return run

    def process_raw_data(self) -> int | None:
        if self.raw_data is None or not self.project.runs:
            raise ValueError("No raw histogram data and RUN configuration are available")
        run = self.project.runs[0]
        alpha = 1.0
        tokens = run.tokens("alpha")
        if tokens:
            try:
                alpha = self.project.parameter_values()[int(float(tokens[0]))]
            except (ValueError, KeyError) as exc:
                raise ValueError("alpha must reference an existing fit parameter") from exc
        self.data = prepare_run_data(self.raw_data, run, alpha=alpha)
        if not isinstance(self.data, CountSpectrum) or run.tokens("norm"):
            return None
        index = max((parameter.index for parameter in self.project.parameters), default=0) + 1
        background = float(self.data.metadata.get("background_per_packed_bin", 0.0))
        estimate = max(float(np.max(self.data.counts)) - background, 1.0)
        self.project.parameters.append(
            Parameter(index, "N0", estimate, max(np.sqrt(estimate), 1.0), 0, None))
        run.options["norm"] = [str(index)]
        return index

