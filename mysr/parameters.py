"""Parse and format the editable free-text parameter block."""
from __future__ import annotations

import math
import re
from collections.abc import Iterable

from .models import Parameter


class ParameterTextError(ValueError):
    """A parameter block contains an invalid line."""


def _error(line_number: int, message: str) -> ParameterTextError:
    return ParameterTextError(f"Parameter line {line_number}: {message}")


def _number(token: str, line_number: int, field: str) -> float:
    try:
        value = float(token)
    except ValueError as exc:
        raise _error(line_number, f"{field} must be a number, got {token!r}") from exc
    if not math.isfinite(value):
        raise _error(line_number, f"{field} must be finite")
    return value


def _boundary(token: str, line_number: int, field: str) -> float | None:
    return None if token.casefold() == "none" else _number(token, line_number, field)


def parse_parameter_text(
    text: str,
    existing: Iterable[Parameter] = (),
) -> list[Parameter]:
    """Parse ``number name value step min max`` lines, ignoring comments/blanks."""
    previous = {parameter.index: parameter for parameter in existing}
    parameters = []
    seen = {}
    for line_number, raw in enumerate(text.splitlines(), 1):
        body = raw.split("#", 1)[0].strip()
        if not body:
            continue
        fields = body.split()
        if len(fields) != 6:
            raise _error(
                line_number,
                "expected 6 fields: number name value step min max",
            )
        try:
            index = int(fields[0])
        except ValueError as exc:
            raise _error(line_number, f"number must be an integer, got {fields[0]!r}") from exc
        if index < 1:
            raise _error(line_number, "number must be positive")
        if index in seen:
            raise _error(line_number, f"duplicate number {index} (first used on line {seen[index]})")
        name = fields[1]
        value = _number(fields[2], line_number, "value")
        step = _number(fields[3], line_number, "step")
        if step < 0:
            raise _error(line_number, "step cannot be negative")
        lower = _boundary(fields[4], line_number, "minimum")
        upper = _boundary(fields[5], line_number, "maximum")
        if lower is not None and upper is not None and lower > upper:
            raise _error(line_number, "minimum exceeds maximum")
        old = previous.get(index)
        parameters.append(Parameter(
            index=index,
            name=name,
            value=value,
            step=step,
            lower=lower,
            upper=upper,
            pos_error=old.pos_error if old is not None else "none",
        ))
        seen[index] = line_number
    if not parameters:
        raise ParameterTextError("The parameter block contains no parameters")
    return parameters


DEFAULT_SIGNIFICANT_DIGITS = 4
_COLUMN_WIDTHS = (8, 20, 16, 16, 16, 0)


def _format_number(value: float | None, significant_digits: int) -> str:
    return "none" if value is None else format(float(value), f".{significant_digits}g")


def _columns(fields: Iterable[object]) -> str:
    result = []
    for field, width in zip(fields, _COLUMN_WIDTHS):
        text = str(field)
        result.append(text.ljust(width) if width else text)
    return "".join(result).rstrip()


def _format_header() -> str:
    return "# " + _columns(("number", "name", "value", "step", "min", "max"))


def _format_line(parameter: Parameter, significant_digits: int) -> str:
    return "  " + _columns((
        parameter.index,
        parameter.name,
        _format_number(parameter.value, significant_digits),
        _format_number(parameter.step, significant_digits),
        _format_number(parameter.lower, significant_digits),
        _format_number(parameter.upper, significant_digits),
    ))


def format_parameter_text(
    parameters: Iterable[Parameter],
    template: str = "",
    significant_digits: int = DEFAULT_SIGNIFICANT_DIGITS,
) -> str:
    """Format parameters while retaining template comments and blank lines."""
    if not 1 <= significant_digits <= 15:
        raise ValueError("Significant digits must be between 1 and 15")
    ordered = list(parameters)
    by_index = {parameter.index: parameter for parameter in ordered}
    if not template:
        lines = [_format_header()]
        lines.extend(_format_line(parameter, significant_digits) for parameter in ordered)
        return "\n".join(lines)

    result = []
    written = set()
    header_fields = ["number", "name", "value", "step", "min", "max"]
    has_header = any(
        raw.partition("#")[1] and raw.partition("#")[2].split() == header_fields
        for raw in template.splitlines()
    )
    for raw in template.splitlines():
        body, separator, comment = raw.partition("#")
        if separator and comment.split() == header_fields:
            result.append(_format_header())
            continue
        match = re.match(r"^(\s*)(\d+)\b", body)
        if match and int(match.group(2)) in by_index:
            if not has_header:
                result.append(_format_header())
                has_header = True
            index = int(match.group(2))
            suffix = ""
            if separator:
                suffix = "  #" + comment
            result.append(_format_line(by_index[index], significant_digits) + suffix)
            written.add(index)
        else:
            result.append(raw)

    missing = [parameter for parameter in ordered if parameter.index not in written]
    if missing:
        if result and result[-1].strip():
            result.append("")
        result.extend(_format_line(parameter, significant_digits) for parameter in missing)
    return "\n".join(result)


def renumber_parameter_text(text: str, *, first: int = 1) -> str:
    """Number every non-comment parameter line sequentially in visual order."""
    if first < 1:
        raise ValueError("The first parameter number must be positive")
    result = []
    next_index = first
    for line_number, raw in enumerate(text.splitlines(), 1):
        body, separator, comment = raw.partition("#")
        if not body.strip():
            result.append(raw)
            continue
        fields = body.split()
        if len(fields) != 6:
            raise _error(
                line_number,
                "expected 6 fields before parameters can be renumbered",
            )
        fields[0] = str(next_index)
        suffix = "  #" + comment if separator else ""
        result.append(" ".join(fields) + suffix)
        next_index += 1
    if next_index == first:
        raise ParameterTextError("The parameter block contains no parameters")
    return "\n".join(result)
