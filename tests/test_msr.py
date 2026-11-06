from pathlib import Path
import numpy as np
import pytest
from mysr.msr import read_msr
from mysr.models import Project
from mysr.parameters import parse_parameter_text
from mysr.core.theory import evaluate_theory

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
    assert "FOURIER" in p.msr_sections
    assert "STATISTIC" in p.msr_sections
    a = evaluate_theory(p.theory, [0, 1, 8], p.parameter_values(), p.functions_text)
    # All but the KT component have zero amplitude in the provided example.
    assert np.isclose(a[0], p.parameters[17].value)
    assert a[1] < a[0]


def test_native_roundtrip_preserves_source(tmp_path):
    p = read_msr(MSR)
    p.custom_python = "# example code"
    dest = tmp_path / "experiment.mysr"
    p.save(dest)
    loaded = Project.load(dest)
    assert loaded.parameters == p.parameters
    assert loaded.parameter_text == p.parameter_text
    assert loaded.theory == p.theory
    assert loaded.source_msr == MSR.read_text()
    assert loaded.msr_sections == p.msr_sections
    assert loaded.custom_python == "# example code"


def test_legacy_project_extension_is_rejected(tmp_path):
    project = read_msr(MSR)
    with pytest.raises(ValueError, match=r"\.mysr extension"):
        project.save(tmp_path / "experiment.pymusr")
