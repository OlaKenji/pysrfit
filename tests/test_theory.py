import numpy as np
import pytest
from pysrfit.core.functions import (
    REGISTRY,
    activate_user_code,
    dynamic_gaussian_kt_lf,
    gaussian_kt,
    gaussian_kt_lf,
)
from pysrfit.core.theory import evaluate_theory


def test_theory_additive_multiplicative_units_and_aliases():
    t = np.array([0., 1.])
    vals = {1: 0.2, 2: 0.1, 3: 0., 4: 0.5, 5: 0.05}
    theory = "a 1\nse 2\ntf 3 4\n+\nasymmetry 5"
    out = evaluate_theory(theory, t, vals)
    assert np.allclose(out, 0.2 * np.exp(-0.1 * t) * np.cos(np.pi * t) + 0.05)


def test_auxiliary_expression_and_map():
    vals = {1: 0.2, 2: 50., 3: 0.}
    out = evaluate_theory("asymmetry map1\nTFieldCos 3 fun1", [0, 1], vals,
                          "fun1 = gamma_mu * par2", mapping=[1])
    assert np.isclose(out[0], 0.2)
    assert np.isclose(out[1], 0.2 * np.cos(2 * np.pi * 0.0135538817 * 50))


def test_malicious_functions_not_executed():
    with pytest.raises(ValueError, match="Unsupported expression"):
        evaluate_theory("asymmetry fun1", [0., 1.], {1: 1},
                        "fun1 = __import__('os').system('echo danger')")


def test_custom_models_work_without_compilation():
    source = '''\nfrom pysrfit.core.functions import register_function\nimport numpy as np\n@register_function("sampleStretched", 2, custom=True)\ndef fn(t, rate, beta):\n    return np.exp(-(rate * np.asarray(t))**beta)\n'''
    activate_user_code(source)
    try:
        out = evaluate_theory("sampleStretched 1 2", [0., 2.], {1: .4, 2: 1.5})
        assert np.allclose(out, np.exp(-(.4 * np.array([0, 2]))**1.5))
    finally:
        REGISTRY.pop("samplestretched", None)


def test_unknown_theory_is_rejected():
    with pytest.raises(ValueError, match="Unknown THEORY"):
        evaluate_theory("fictitiousRelax 1 2", [0, 1], {1: 1., 2: 2.})


def test_static_gaussian_lf_kt_has_correct_limits_and_integral():
    time = np.linspace(0.0, 4.0, 81)
    delta = 0.3
    assert np.allclose(gaussian_kt_lf(time, 0.0, delta), gaussian_kt(time, delta))
    assert np.allclose(gaussian_kt_lf(time, 0.5, 0.0), 1.0)
    assert np.min(gaussian_kt_lf(time, 5.0, delta)) > 0.99

    frequency = 0.5
    selected_time = 1.25
    integration_time = np.linspace(0.0, selected_time, 200_001)
    omega = 2 * np.pi * frequency
    integrand = (
        np.exp(-0.5 * (delta * integration_time) ** 2)
        * np.sin(omega * integration_time)
    )
    integral = np.sum(
        0.5 * np.diff(integration_time) * (integrand[:-1] + integrand[1:]))
    expected = (
        1
        - 2 * (delta / omega) ** 2
        * (1 - np.exp(-0.5 * (delta * selected_time) ** 2)
           * np.cos(omega * selected_time))
        + 2 * delta**4 / omega**3 * integral
    )
    assert gaussian_kt_lf([selected_time], frequency, delta)[0] == pytest.approx(
        expected, abs=2e-7)


def test_dynamic_gaussian_lf_kt_limits_and_theory_aliases():
    time = np.linspace(0.0, 5.0, 101)
    static = gaussian_kt_lf(time, 0.2, 0.3)
    assert np.allclose(dynamic_gaussian_kt_lf(time, 0.2, 0.3, 0.0), static)
    dynamic = dynamic_gaussian_kt_lf(time, 0.2, 0.3, 0.1)
    fast = dynamic_gaussian_kt_lf(time, 0.2, 0.3, 2.0)
    assert dynamic[0] == pytest.approx(1.0)
    assert fast[0] == pytest.approx(1.0)
    assert np.all(np.isfinite(dynamic))
    assert np.all(np.isfinite(fast))

    values = {1: 0.2, 2: 0.3, 3: 0.1}
    assert np.allclose(
        evaluate_theory("sgktlf 1 2", time, values),
        evaluate_theory("statGssKTLF 1 2", time, values),
    )
    assert np.allclose(
        evaluate_theory("dgktlf 1 2 3", time, values),
        evaluate_theory("dynGssKTLF 1 2 3", time, values),
    )
