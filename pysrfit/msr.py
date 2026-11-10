"""Conservative musrfit .msr importer for pysrfit; unsupported sections are retained."""
from __future__ import annotations
from pathlib import Path
import re
from .models import Parameter, Project, RunConfig
from .parameters import format_parameter_text

HEADERS = {"TITLE", "FITPARAMETER", "THEORY", "FUNCTIONS", "GLOBAL",
           "RUN", "COMMANDS", "FOURIER", "PLOT", "STATISTIC"}


def _body(line: str) -> str:
    """Remove common MSR inline comments, not arbitrary expression syntax."""
    return line.split("#", 1)[0].split("(", 1)[0].strip()


def _split_sections(raw: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {"PREAMBLE": []}
    section = "PREAMBLE"
    for line in raw.splitlines():
        stripped = line.strip()
        # musrfit section headers are uppercase. Keep this case-sensitive so
        # directives such as FOURIER's ``plot POWER`` are not mistaken for a
        # new PLOT section.
        first = stripped.split(None, 1)[0] if stripped else ""
        if first in HEADERS and not stripped.startswith("#"):
            section = first
            sections.setdefault(section, [])
            # Preserve RUN and PLOT etc header arguments within their blocks.
            sections[section].append(line)
        else:
            sections[section].append(line)
    return sections


def parse_msr(text: str, *, source: str | Path | None = None) -> Project:
    sections = _split_sections(text)
    title = next((line.strip() for line in sections["PREAMBLE"] if line.strip() and not line.lstrip().startswith("#")), "Untitled")
    parameters: list[Parameter] = []
    for line in sections.get("FITPARAMETER", [])[1:]:
        parts = _body(line).split()
        if not parts or not re.fullmatch(r"\d+", parts[0]):
            continue
        if len(parts) < 4:
            raise ValueError(f"Incomplete FITPARAMETER line: {line}")
        def boundary(i: int):
            if i >= len(parts) or parts[i].lower() == "none":
                return None
            return float(parts[i])
        parameters.append(Parameter(
            index=int(parts[0]), name=parts[1], value=float(parts[2]),
            step=float(parts[3]), pos_error=parts[4] if len(parts) > 4 else "none",
            lower=boundary(5), upper=boundary(6)))
    runs: list[RunConfig] = []
    current: RunConfig | None = None
    for line in sections.get("RUN", []):
        bits = _body(line).split()
        if not bits:
            continue
        if bits[0].upper() == "RUN":
            current = RunConfig(source=bits[1] if len(bits) > 1 else "",
                                instrument=bits[2] if len(bits) > 2 else "",
                                facility=bits[3] if len(bits) > 3 else "",
                                format_hint=bits[4] if len(bits) > 4 else "")
            runs.append(current)
        elif current is not None:
            current.options[bits[0].lower()] = bits[1:]
            if bits[0].lower() == "fittype" and len(bits) > 1:
                current.fittype = int(bits[1])
    def block(name):
        return "\n".join(sections.get(name, [])[1:]).strip()
    parameter_template = "\n".join(sections.get("FITPARAMETER", [])[1:]).strip("\n")
    parameter_template = "\n".join(
        "# number name value step min max"
        if "pos_error" in line.casefold() and "boundar" in line.casefold()
        else line
        for line in parameter_template.splitlines()
    )
    project = Project(title=title, parameters=parameters,
                      # Import without losing digits from the source. The GUI's
                      # four-digit default applies when the user formats or fits.
                      parameter_text=format_parameter_text(
                          parameters, parameter_template, significant_digits=15),
                      theory=block("THEORY"),
                      functions_text=block("FUNCTIONS"), runs=runs,
                      source_msr=text, msr_sections=sections)
    if source is not None:
        project.location = Path(source).resolve()
    if "PLOT" in sections:
        for line in sections["PLOT"]:
            items = _body(line).split()
            if items and items[0].lower() == "range" and len(items) >= 3:
                start, stop = float(items[1]), float(items[2])
                if stop <= start:
                    raise ValueError("PLOT time range must have increasing limits")
                project.display.update(t_start=start, t_stop=stop)
                if len(items) >= 5:
                    y_min, y_max = float(items[3]), float(items[4])
                    if y_max > y_min:
                        project.display.update(
                            y_min=y_min, y_max=y_max, y_auto=False)
                break
        for line in sections["PLOT"]:
            items = _body(line).split()
            if items and items[0].lower() == "view_packing" and len(items) >= 2:
                packing = int(float(items[1]))
                if packing < 1:
                    raise ValueError("PLOT view_packing must be positive")
                project.display["view_packing"] = packing
                break
    return project


def read_msr(path: str | Path) -> Project:
    path = Path(path)
    return parse_msr(path.read_text(encoding="utf-8-sig"), source=path)
