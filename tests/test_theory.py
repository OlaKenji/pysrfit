import numpy as np
import pytest
from pysrfit.core.functions import (
    REGISTRY,
    activate_user_code,
    bessel,
    dynamic_gaussian_kt_lf,
    dynamic_lorentzian_kt_lf,
    gaussian_kt,
    gaussian_kt_lf,
    lorentzian_kt,
    lorentzian_kt_lf,
    stretched_expo,
)
from pysrfit.core.theory import evaluate_theory


def test_theory_additive_and_multiplicative_components():
    t = np.array([0., 1.])
    vals = {1: 0.2, 2: 0.1, 3: 0., 4: 0.5, 5: 0.05}
    theory = "asymmetry 1\nsimplExpo 2\nTFieldCos 3 4\n+\nasymmetry 5"
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


def test_theory_abbreviations_are_not_supported():
    with pytest.raises(ValueError, match="Unknown THEORY"):
        evaluate_theory("a 1", [0, 1], {1: 0.2})


def test_simple_gaussian_and_bessel():
    time = np.array([0.0, 1.0])
    sigma = 0.3
    assert np.allclose(
        evaluate_theory("simpleGss 1", time, {1: sigma}),
        np.exp(-0.5 * (sigma * time) ** 2),
    )
    frequency_at_first_zero = 2.404825557695773 / (2.0 * np.pi)
    values = bessel(time, 0.0, frequency_at_first_zero)
    assert values[0] == pytest.approx(1.0)
    assert values[1] == pytest.approx(0.0, abs=1e-8)
    assert bessel([1.0], 0.0, 1.0 / (2.0 * np.pi))[0] == pytest.approx(
        0.7651976865579666, abs=1e-8)


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


def test_dynamic_gaussian_lf_kt_limits_and_theory_names():
    time = np.linspace(0.0, 5.0, 101)
    static = gaussian_kt_lf(time, 0.2, 0.3)
    assert np.allclose(dynamic_gaussian_kt_lf(time, 0.2, 0.3, 0.0), static)
    dynamic = dynamic_gaussian_kt_lf(time, 0.2, 0.3, 0.1)
    fast = dynamic_gaussian_kt_lf(time, 0.2, 0.3, 2.0)
    assert dynamic[0] == pytest.approx(1.0)
    assert fast[0] == pytest.approx(1.0)
    assert np.all(np.isfinite(dynamic))
    assert np.all(np.isfinite(fast))

    assert np.allclose(
        evaluate_theory(
            "dynGssKTLF 1 2 3", time, {1: 0.2, 2: 0.3, 3: 0.1}),
        dynamic,
    )


def test_stretched_exponential():
    time = np.linspace(0.0, 5.0, 51)
    rate = 0.3

    assert np.allclose(stretched_expo(time, rate, 1.0), np.exp(-rate * time))
    assert np.allclose(
        stretched_expo(time, rate, 2.0), np.exp(-(rate * time) ** 2))
    assert np.allclose(
        evaluate_theory("generExpo 1 2", time, {1: rate, 2: 0.7}),
        stretched_expo(time, rate, 0.7),
    )


def test_static_lorentzian_kt_lf_limits_and_integral():
    time = np.linspace(0.0, 5.0, 101)
    damping = 0.3
    assert np.allclose(
        lorentzian_kt_lf(time, 0.0, damping),
        lorentzian_kt(time, damping),
    )
    assert np.allclose(lorentzian_kt_lf(time, 0.5, 0.0), 1.0)
    assert np.min(lorentzian_kt_lf(time, 5.0, damping)) > 0.97

    frequency = 0.5
    selected_time = 1.25
    integration_time = np.linspace(0.0, selected_time, 200_001)
    omega = 2.0 * np.pi * frequency
    argument = omega * integration_time
    j0 = np.ones_like(argument)
    j0[1:] = np.sin(argument[1:]) / argument[1:]
    integrand = np.exp(-damping * integration_time) * j0
    integral = np.sum(
        0.5 * np.diff(integration_time)
        * (integrand[:-1] + integrand[1:]))
    x = omega * selected_time
    j0_value = np.sin(x) / x
    j1_value = (np.sin(x) - x * np.cos(x)) / x**2
    ratio = damping / omega
    expected = (
        1.0
        - ratio * j1_value * np.exp(-damping * selected_time)
        - ratio**2 * (j0_value * np.exp(-damping * selected_time) - 1.0)
        - damping * (1.0 + ratio**2) * integral
    )
    assert lorentzian_kt_lf(
        [selected_time], frequency, damping)[0] == pytest.approx(
            expected, abs=2e-7)


def test_dynamic_lorentzian_lf_kt_limits_and_theory_names():
    time = np.linspace(0.0, 5.0, 101)
    static = lorentzian_kt_lf(time, 0.2, 0.3)
    assert np.allclose(
        dynamic_lorentzian_kt_lf(time, 0.2, 0.3, 0.0), static)
    dynamic = dynamic_lorentzian_kt_lf(time, 0.2, 0.3, 0.1)
    fast = dynamic_lorentzian_kt_lf(time, 0.2, 0.3, 2.0)
    assert dynamic[0] == pytest.approx(1.0)
    assert fast[0] == pytest.approx(1.0)
    assert np.all(np.isfinite(dynamic))
    assert np.all(np.isfinite(fast))

    values = {1: 0.2, 2: 0.3, 3: 0.1}
    assert np.allclose(
        evaluate_theory("statExpKT 2", time, values),
        lorentzian_kt(time, 0.3),
    )
    assert np.allclose(
        evaluate_theory("statExpKTLF 1 2", time, values),
        static,
    )
    assert np.allclose(
        evaluate_theory("dynExpKTLF 1 2 3", time, values),
        dynamic,
    )


def test_two_state_zero_rate_is_weighted_static_mixture():
    time = np.linspace(0.0, 4.0, 81)
    values = {
        1: 0.0,
        2: 0.0,
        3: 0.3,
        4: 0.0,
        5: 0.7,
        6: 0.25,
        7: 0.12,
    }
    theory = (
        "twoState 1 2 3 | TFieldCos 4 5 * simplExpo 7 | statGssKt 6"
    )
    expected_first = np.cos(2.0 * np.pi * values[5] * time) * np.exp(
        -values[7] * time)
    expected_second = gaussian_kt(time, values[6])
    expected = values[3] * expected_first + (1.0 - values[3]) * expected_second
    assert np.allclose(evaluate_theory(theory, time, values), expected)


def test_two_state_conserves_polarization_for_constant_static_states():
    time = np.linspace(0.0, 5.0, 101)
    values = {1: 0.7, 2: 1.3, 3: 0.37, 4: 0.0}
    result = evaluate_theory(
        "twoState 1 2 3 | simplExpo 4 | simplExpo 4", time, values)
    assert np.allclose(result, 1.0, atol=2e-5)


def test_two_state_identical_static_functions_reduce_to_static_function():
    time = np.linspace(0.0, 4.0, 81)
    values = {1: 2.0, 2: 0.4, 3: 0.65, 4: 0.23}
    result = evaluate_theory(
        "twoState 1 2 3 | simplExpo 4 | simplExpo 4", time, values)
    assert np.allclose(result, np.exp(-values[4] * time), atol=2e-5)


def test_two_state_accepts_custom_static_functions():
    source = '''\nfrom pysrfit.core.functions import register_function\nimport numpy as np\n@register_function("sampleStatic", 1, custom=True)\ndef fn(t, frequency):\n    return np.cos(2 * np.pi * frequency * np.asarray(t))\n'''
    activate_user_code(source)
    try:
        time = np.linspace(0.0, 2.0, 41)
        values = {1: 0.0, 2: 0.0, 3: 0.4, 4: 0.7, 5: 0.2}
        result = evaluate_theory(
            "twoState 1 2 3 | sampleStatic 4 | sampleStatic 5",
            time,
            values,
        )
        expected = (
            values[3] * np.cos(2.0 * np.pi * values[4] * time)
            + (1.0 - values[3]) * np.cos(2.0 * np.pi * values[5] * time)
        )
        assert np.allclose(result, expected)
    finally:
        REGISTRY.pop("samplestatic", None)


def test_two_state_reports_incomplete_composite_syntax():
    with pytest.raises(ValueError, match="twoState syntax"):
        evaluate_theory(
            "twoState 1 2 3 | statGssKt 4",
            [0.0, 1.0],
            {1: 0.1, 2: 0.1, 3: 0.5, 4: 0.2},
        )
