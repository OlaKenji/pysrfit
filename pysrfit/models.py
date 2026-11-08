"""Format-agnostic persistent scientific project model for pysrfit."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any
import json
from pathlib import Path

SCHEMA_VERSION = 1
NATIVE_PROJECT_SUFFIX = ".pysrfit"
DEFAULT_PACKING = 20


def default_fourier_settings() -> dict[str, Any]:
    return {
        "window": "Hann",
        "subtract_mean": True,
        "zero_padding": 2,
        "phase_deg": 0.0,
        "output": "Magnitude",
        "normalize": True,
        "units": "MHz",
        "frequency_min": 0.0,
        "frequency_max": 100.0,
    }


def _require_native_project_suffix(path: Path) -> None:
    if path.suffix.lower() != NATIVE_PROJECT_SUFFIX:
        raise ValueError(
            f"Native PySRFit projects must use the {NATIVE_PROJECT_SUFFIX} extension: {path}"
        )


@dataclass
class Parameter:
    index: int
    name: str
    value: float
    step: float = 0.01
    lower: float | None = None
    upper: float | None = None
    pos_error: str = "none"

    @property
    def fixed(self) -> bool:
        return self.step == 0.0


@dataclass
class RunConfig:
    source: str = ""
    label: str = ""
    plot_enabled: bool = True
    instrument: str = ""
    facility: str = ""
    format_hint: str = ""
    fittype: int = 2
    options: dict[str, list[str]] = field(default_factory=dict)
    file_metadata: dict[str, Any] = field(default_factory=dict)
    user_metadata: dict[str, str] = field(default_factory=dict)

    def tokens(self, key: str) -> list[str]:
        return self.options.get(key.lower(), [])


@dataclass
class FitRecord:
    """Last completed fit, persisted independently from editable parameters."""

    valid: bool
    chi2: float | None = None
    ndf: int | None = None
    values: dict[int, float] = field(default_factory=dict)
    errors: dict[int, float] = field(default_factory=dict)
    run_chi2: list[float] = field(default_factory=list)
    run_points: list[int] = field(default_factory=list)
    message: str = ""

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> FitRecord:
        return cls(
            valid=bool(obj.get("valid", False)),
            chi2=None if obj.get("chi2") is None else float(obj["chi2"]),
            ndf=None if obj.get("ndf") is None else int(obj["ndf"]),
            values={int(key): float(value) for key, value in obj.get("values", {}).items()},
            errors={int(key): float(value) for key, value in obj.get("errors", {}).items()},
            run_chi2=[float(value) for value in obj.get("run_chi2", [])],
            run_points=[int(value) for value in obj.get("run_points", [])],
            message=str(obj.get("message", "")),
        )


@dataclass
class Project:
    title: str = "Untitled μSR project"
    parameters: list[Parameter] = field(default_factory=list)
    parameter_text: str = ""  # Editable layout with comments and blank lines.
    theory: str = "asymmetry 1\nsimplExpo 2"
    functions_text: str = ""
    runs: list[RunConfig] = field(default_factory=list)
    custom_python: str = ""
    source_msr: str = ""  # Original, unmodified MSR text for audit/round-trip reference
    msr_sections: dict[str, list[str]] = field(default_factory=dict)
    spectrum_path: str = ""  # Fit-ready 2/3-column spectrum, if used
    data_override: str = ""  # Optional raw-run path override
    display: dict[str, Any] = field(default_factory=lambda: {"t_start": 0.0, "t_stop": 8.0})
    fourier: dict[str, Any] = field(default_factory=default_fourier_settings)
    last_fit: FitRecord | None = None
    schema_version: int = SCHEMA_VERSION
    # Runtime only, not serialized. Enables relative file resolution.
    location: Path | None = field(default=None, repr=False, compare=False)

    def parameter_values(self) -> dict[int, float]:
        return {p.index: p.value for p in self.parameters}

    def to_dict(self) -> dict[str, Any]:
        obj = asdict(self)
        obj.pop("location", None)
        return obj

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> Project:
        if obj.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported project schema version {obj.get('schema_version')}")
        allowed = {
            k: v for k, v in obj.items()
            if k not in ("location", "parameters", "runs", "last_fit")
        }
        allowed["parameters"] = [Parameter(**p) for p in obj.get("parameters", [])]
        allowed["runs"] = [RunConfig(**run) for run in obj.get("runs", [])]
        last_fit = obj.get("last_fit")
        allowed["last_fit"] = (
            FitRecord.from_dict(last_fit) if last_fit is not None else None)
        return cls(**allowed)

    def save(self, path: str | Path) -> None:
        dest = Path(path).expanduser().resolve()
        _require_native_project_suffix(dest)
        dest.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self.location = dest

    @classmethod
    def load(cls, path: str | Path) -> Project:
        dest = Path(path).expanduser().resolve()
        _require_native_project_suffix(dest)
        obj = cls.from_dict(json.loads(dest.read_text(encoding="utf-8")))
        obj.location = dest
        return obj

    def resolve(self, reference: str) -> Path:
        path = Path(reference).expanduser()
        return path.resolve() if path.is_absolute() else ((self.location.parent if self.location else Path.cwd()) / path).resolve()


def demo_project() -> Project:
    return Project(title="Demo: exponential + KT", parameters=[
        Parameter(1, "Asymmetry", 0.23, 0.005, 0, 0.33),
        Parameter(2, "Lambda", 0.15, 0.01, 0, 5),
        Parameter(3, "DeltaKT", 0.12, 0.005, 0, 2),
    ], theory="asymmetry 1\nsimplExpo 2\nstatGssKt 3")
