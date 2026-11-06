import numpy as np
import pytest
from mysr.core.functions import activate_user_code, REGISTRY
from mysr.core.theory import evaluate_theory


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
    source = '''\nfrom mysr.core.functions import register_function\nimport numpy as np\n@register_function("sampleStretched", 2, custom=True)\ndef fn(t, rate, beta):\n    return np.exp(-(rate * np.asarray(t))**beta)\n'''
    activate_user_code(source)
    try:
        out = evaluate_theory("sampleStretched 1 2", [0., 2.], {1: .4, 2: 1.5})
        assert np.allclose(out, np.exp(-(.4 * np.array([0, 2]))**1.5))
    finally:
        REGISTRY.pop("samplestretched", None)


def test_unknown_theory_is_rejected():
    with pytest.raises(ValueError, match="Unknown THEORY"):
        evaluate_theory("fictitiousRelax 1 2", [0, 1], {1: 1., 2: 2.})
