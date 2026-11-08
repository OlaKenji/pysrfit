from pathlib import Path

import pytest

from pysrfit.run_numbers import (
    available_run_sources,
    infer_run_number,
    metadata_run_number,
    neighboring_run_source,
    path_for_run_number,
)


def test_metadata_run_number_supports_nested_common_keys():
    assert metadata_run_number({"run_number": 3092}) == 3092
    assert metadata_run_number({"file_header": {"Run Number": "42"}}) == 42
    assert metadata_run_number({"temperature": 5}) is None


def test_psi_filename_replaces_run_not_year_and_preserves_padding():
    source = Path("deltatbc_tdc_gps_2025_003092.root")

    target = path_for_run_number(source, 3093, {"run_number": 3092})

    assert target == Path("deltatbc_tdc_gps_2025_003093.root")
    assert infer_run_number(source, {"run_number": 3092}) == 3092


def test_filename_fallback_uses_final_numeric_token():
    source = Path("gps_2025_0042.nxs")
    assert infer_run_number(source) == 42
    assert path_for_run_number(source, 43) == Path("gps_2025_0043.nxs")


def test_filename_without_numeric_token_is_rejected():
    with pytest.raises(ValueError, match="Cannot identify"):
        path_for_run_number(Path("current.root"), 2)


def test_available_navigation_skips_missing_runs_and_unrelated_files(tmp_path):
    sources = [
        tmp_path / "gps_2025_003092.root",
        tmp_path / "gps_2025_003094.root",
        tmp_path / "gps_2025_003100.root",
    ]
    for source in sources:
        source.write_text("data")
    (tmp_path / "other_003095.root").write_text("other")
    (tmp_path / "gps_2025_003095.dat").write_text("wrong extension")

    assert available_run_sources(sources[1], {"run_number": 3094}) == [
        (3092, sources[0]), (3094, sources[1]), (3100, sources[2])]
    assert neighboring_run_source(sources[1], -1, {"run_number": 3094}) == (
        3092, sources[0])
    assert neighboring_run_source(sources[1], 1, {"run_number": 3094}) == (
        3100, sources[2])
    assert neighboring_run_source(sources[0], -1) is None
    assert neighboring_run_source(sources[2], 1) is None
