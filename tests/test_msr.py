from pathlib import Path
import numpy as np
import pytest
from pysrfit.msr import read_msr
from pysrfit.models import Project
from pysrfit.parameters import parse_parameter_text
from pysrfit.core.theory import evaluate_theory

MSR = Path(__file__).resolve().parents[1] / "examples" / "ZF44K.msr"


def test_msr_import_and_preview():
    p = read_msr(MSR)
    assert p.title == "ZF,45K"
    assert len(p.parameters) == 23
    assert p.parameters[0].name == "alpha"
    assert p.parameters[0].fixed
    assert p.parameters[17].index == 18
    assert parse_parameter_text(p.parameter_text) == p.parameters
    assert len(p.runs) == 1
    assert p.runs[0].source == "./Data/deltat_tdc_gps_2651"
    assert p.runs[0].format_hint == "PSI-BIN"
    assert p.runs[0].tokens("forward") == ["3"]
    assert p.runs[0].tokens("backward") == ["4"]
    assert p.runs[0].tokens("t0") == ["401.0", "403.0"]
    assert p.display["t_start"] == pytest.approx(0.002)
    assert p.display["t_stop"] == pytest.approx(8.0)
    assert p.display["y_min"] == pytest.approx(0.0)
    assert p.display["y_max"] == pytest.approx(0.3)
    assert not p.display["y_auto"]
    assert p.display["view_packing"] == 200
    assert any(
        line.split()[:2] == ["plot", "POWER"]
        for line in p.msr_sections["FOURIER"]
    )
    assert "FOURIER" in p.msr_sections
    assert "STATISTIC" in p.msr_sections
    a = evaluate_theory(p.theory, [0, 1, 8], p.parameter_values(), p.functions_text)
    # All but the KT component have zero amplitude in the provided example.
    assert np.isclose(a[0], p.parameters[17].value)
    assert a[1] < a[0]


def test_native_roundtrip_preserves_source(tmp_path):
    p = read_msr(MSR)
    p.custom_python = "# example code"
    p.runs[0].file_metadata = {"temperature": "5 K", "run_number": 2651}
    p.runs[0].user_metadata = {"comment": "field cooled"}
    dest = tmp_path / "experiment.pysrfit"
    p.save(dest)
    loaded = Project.load(dest)
    assert loaded.parameters == p.parameters
    assert loaded.parameter_text == p.parameter_text
    assert loaded.theory == p.theory
    assert loaded.source_msr == MSR.read_text()
    assert loaded.msr_sections == p.msr_sections
    assert loaded.custom_python == "# example code"
    assert loaded.fourier == p.fourier
    assert loaded.runs[0].file_metadata == p.runs[0].file_metadata
    assert loaded.runs[0].user_metadata == p.runs[0].user_metadata


@pytest.mark.parametrize("suffix", [".mysr", ".pymusr"])
def test_non_pysrfit_project_extensions_are_rejected(tmp_path, suffix):
    project = read_msr(MSR)
    with pytest.raises(ValueError, match=r"\.pysrfit extension"):
        project.save(tmp_path / f"experiment{suffix}")
