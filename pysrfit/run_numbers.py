"""Infer run numbers and resolve neighboring data-file paths."""
from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Any


_RUN_KEYS = {"run", "runnumber", "runno"}
_DIGITS = re.compile(r"\d+")


def _number(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or not numeric.is_integer() or numeric < 0:
        return None
    return int(numeric)


def metadata_run_number(metadata: dict[str, Any]) -> int | None:
    """Return the first run-number-like value in nested file metadata."""
    for key, value in metadata.items():
        normalized = re.sub(r"[^a-z0-9]", "", str(key).casefold())
        if normalized in _RUN_KEYS:
            number = _number(value)
            if number is not None:
                return number
    for value in metadata.values():
        if isinstance(value, dict):
            number = metadata_run_number(value)
            if number is not None:
                return number
    return None


def _run_token(path: Path, metadata: dict[str, Any] | None = None):
    matches = list(_DIGITS.finditer(path.stem))
    if not matches:
        return None
    current = metadata_run_number(metadata or {})
    if current is not None:
        matching = [match for match in matches if int(match.group()) == current]
        if matching:
            return matching[-1]
    # Facility filenames normally place the run number after year/instrument
    # tokens. Facility-specific resolvers can replace this fallback later.
    return matches[-1]


def infer_run_number(
    path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> int | None:
    """Infer a run number from metadata, then the final filename digit token."""
    number = metadata_run_number(metadata or {})
    if number is not None:
        return number
    token = _run_token(Path(path), metadata)
    return int(token.group()) if token is not None else None


def path_for_run_number(
    path: str | Path,
    run_number: int,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Replace the source filename's run token, preserving its zero-padding."""
    target = _number(run_number)
    if target is None:
        raise ValueError("Run number must be a non-negative integer")
    source = Path(path)
    token = _run_token(source, metadata)
    if token is None:
        raise ValueError(f"Cannot identify a run number in {source.name!r}")
    replacement = str(target).zfill(len(token.group()))
    stem = source.stem
    new_stem = stem[:token.start()] + replacement + stem[token.end():]
    return source.with_name(new_stem + source.suffix)


def available_run_sources(
    path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> list[tuple[int, Path]]:
    """List sibling files matching the source's run-number filename pattern."""
    source = Path(path)
    token = _run_token(source, metadata)
    if token is None or not source.parent.is_dir():
        return []
    prefix = re.escape(source.stem[:token.start()])
    suffix = re.escape(source.stem[token.end():])
    pattern = re.compile(rf"^{prefix}(\d+){suffix}$")
    matches = []
    for candidate in source.parent.iterdir():
        if not candidate.is_file() or candidate.suffix != source.suffix:
            continue
        match = pattern.fullmatch(candidate.stem)
        if match:
            matches.append((int(match.group(1)), candidate))
    return sorted(matches, key=lambda item: (item[0], item[1].name))


def neighboring_run_source(
    path: str | Path,
    direction: int,
    metadata: dict[str, Any] | None = None,
) -> tuple[int, Path] | None:
    """Return the nearest available lower or higher run with the same pattern."""
    if direction not in (-1, 1):
        raise ValueError("Run navigation direction must be -1 or 1")
    source = Path(path)
    token = _run_token(source, metadata)
    if token is None:
        return None
    current = int(token.group())
    available = available_run_sources(source, metadata)
    if direction < 0:
        lower = [item for item in available if item[0] < current]
        return lower[-1] if lower else None
    return next((item for item in available if item[0] > current), None)
