import numpy as np
import pytest
from pysrfit.msr import read_msr
from pysrfit.models import Project
from pysrfit.parameters import parse_parameter_text
from pysrfit.core.theory import evaluate_theory

MSR_TEXT = """ZF,45K
FITPARAMETER
# number name value step pos_error lower upper
1 alpha 1.0 0 none none none
2 p2 0.0 0 none none none
3 p3 0.0 0 none none none
4 p4 0.0 0 none none none
5 p5 0.0 0 none none none
6 p6 0.0 0 none none none
7 p7 0.0 0 none none none
8 p8 0.0 0 none none none
9 p9 0.0 0 none none none
10 p10 0.0 0 none none none
11 p11 0.0 0 none none none
12 p12 0.0 0 none none none
13 p13 0.0 0 none none none
14 p14 0.0 0 none none none
15 p15 0.0 0 none none none
16 p16 0.0 0 none none none
17 p17 0.0 0 none none none
18 asym 0.2 0.01 none none none
19 delta 0.3 0.01 none none none
20 p20 0.0 0 none none none
21 p21 0.0 0 none none none
22 p22 0.0 0 none none none
23 p23 0.0 0 none none none
THEORY
asymmetry 18
statGssKt 19
RUN ./Data/deltat_tdc_gps_2651 GPS PSI PSI-BIN
fittype 2
forward 3
backward 4
t0 401.0 403.0
FOURIER
plot POWER
PLOT 0
range 0.002 8.0 0.0 0.3
view_packing 200
STATISTIC
chisq 1.0
"""


@pytest.fixture
def msr_path(tmp_path):
    path = tmp_path / "ZF44K.msr"
    path.write_text(MSR_TEXT)
    return path


def test_msr_import_and_preview(msr_path):
    p = read_msr(msr_path)
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


def test_native_roundtrip_preserves_source(tmp_path, msr_path):
    p = read_msr(msr_path)
    p.custom_python = "# example code"
    p.runs[0].file_metadata = {"temperature": "5 K", "run_number": 2651}
    p.runs[0].user_metadata = {"comment": "field cooled"}
    dest = tmp_path / "experiment.pysrfit"
    p.save(dest)
    loaded = Project.load(dest)
    assert loaded.parameters == p.parameters
    assert loaded.parameter_text == p.parameter_text
    assert loaded.theory == p.theory
    assert loaded.source_msr == MSR_TEXT
    assert loaded.msr_sections == p.msr_sections
    assert loaded.custom_python == "# example code"
    assert loaded.fourier == p.fourier
    assert loaded.runs[0].file_metadata == p.runs[0].file_metadata
    assert loaded.runs[0].user_metadata == p.runs[0].user_metadata


@pytest.mark.parametrize("suffix", [".mysr", ".pymusr"])
def test_non_pysrfit_project_extensions_are_rejected(tmp_path, suffix, msr_path):
    project = read_msr(msr_path)
    with pytest.raises(ValueError, match=r"\.pysrfit extension"):
        project.save(tmp_path / f"experiment{suffix}")
