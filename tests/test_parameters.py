import pytest

from mysr.models import Parameter
from mysr.parameters import (
    DEFAULT_SIGNIFICANT_DIGITS,
    ParameterTextError,
    format_parameter_text,
    parse_parameter_text,
    renumber_parameter_text,
)


def test_parameter_text_supports_comments_blank_lines_and_none():
    text = """# Sample
1  asymmetry  0.22  0.005  0  0.33

# Background
4  bgRate  0.01  0  none  none  # fixed
"""
    parameters = parse_parameter_text(text)
    assert [parameter.index for parameter in parameters] == [1, 4]
    assert parameters[0] == Parameter(1, "asymmetry", 0.22, 0.005, 0.0, 0.33)
    assert parameters[1].fixed
    assert parameters[1].lower is None
    assert parameters[1].upper is None


def test_parameter_text_reports_line_number_and_duplicate():
    with pytest.raises(ParameterTextError, match="line 3.*duplicate number 1"):
        parse_parameter_text("1 A 0.2 0.01 0 1\n\n1 B 0.1 0.01 0 1")
    with pytest.raises(ParameterTextError, match="line 2.*expected 6 fields"):
        parse_parameter_text("# heading\n2 incomplete 0.1")


def test_formatting_fit_results_preserves_visual_layout():
    template = """# Sample contribution
1 A 0.2 0.01 0 1

# Background contribution
2 BG 0.1 0 none none  # keep fixed
"""
    updated = [
        Parameter(1, "A", 0.234567, 0.002, 0, 1),
        Parameter(2, "BG", 0.08, 0, None, None),
        Parameter(3, "rate", 0.5, 0.01, 0, 10),
    ]
    text = format_parameter_text(updated, template, significant_digits=8)
    assert "# Sample contribution" in text
    assert "\n\n# Background contribution" in text
    assert "# keep fixed" in text
    assert text.rstrip().endswith("10")
    assert parse_parameter_text(text) == updated


def test_default_parameter_precision_is_four_significant_digits():
    assert DEFAULT_SIGNIFICANT_DIGITS == 4
    text = format_parameter_text([Parameter(1, "A", 0.123456, 0.0123456)])
    assert "0.1235" in text
    assert "0.01235" in text


def test_parameter_columns_are_left_aligned_with_header_and_precision():
    text = format_parameter_text(
        [Parameter(1, "amplitude", 0.123456789, 0.001234567, None, 0.3333333)],
        significant_digits=5,
    )
    header, row = text.splitlines()
    assert header.index("number") == row.index("1")
    assert header.index("name") == row.index("amplitude")
    assert header.index("value") == row.index("0.12346")
    assert "0.0012346" in row
    assert "0.33333" in row


def test_parameter_text_can_be_renumbered_before_numbers_are_valid():
    text = """wrong A 0.2 0.01 0 1

# another component
99 B 0.1 0 none none  # fixed
"""
    renumbered = renumber_parameter_text(text)
    parameters = parse_parameter_text(renumbered)
    assert [parameter.index for parameter in parameters] == [1, 2]
    assert "# another component" in renumbered
    assert "# fixed" in renumbered
    formatted = format_parameter_text(parameters, renumbered)
    assert "# number  name" in formatted
