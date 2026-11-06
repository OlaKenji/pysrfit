"""Model function registry. Add user models via register_function decorator."""
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Callable
import numpy as np

@dataclass(frozen=True)
class FunctionSpec:
    name: str
    arity: int
    function: Callable
    description: str = ""
    custom: bool = False

REGISTRY: dict[str, FunctionSpec] = {}


def register_function(name: str, arity: int, description: str = "", *, custom: bool = False):
    if arity < 0:
        raise ValueError("arity cannot be negative")
    def decorator(fn):
        REGISTRY[name.lower()] = FunctionSpec(name, arity, fn, description, custom=custom)
        return fn
    return decorator


@register_function("asymmetry", 1, "Amplitude prefactor")
def asymmetry(t, a):
    return np.full_like(np.asarray(t, dtype=float), float(a))


@register_function("simplExpo", 1, "Exponential: exp(-λt)")
def simple_expo(t, lam):
    return np.exp(-lam * np.asarray(t))


@register_function("simpleGss", 1, "Gaussian: exp(-σ²t²/2)")
def simple_gss(t, sigma):
    return np.exp(-0.5 * (sigma * np.asarray(t)) ** 2)


@register_function("TFieldCos", 2, "cos(2πνt + phase[deg]); ν in MHz")
def tf_cos(t, phase_deg, freq_mhz):
    return np.cos(2 * np.pi * freq_mhz * np.asarray(t) + np.deg2rad(phase_deg))


@register_function("statGssKt", 1, "Static Gaussian Kubo–Toyabe")
def gaussian_kt(t, delta):
    q = (delta * np.asarray(t)) ** 2
    return (1 + 2 * (1 - q) * np.exp(-0.5 * q)) / 3


ALIASES = {"a": "asymmetry", "se": "simplExpo", "sg": "simpleGss", "tf": "TFieldCos"}

CUSTOM_EXAMPLE = '''# This code runs only when you click "Activate code".
# You can write your own Python relaxation functions without compilation.
import numpy as np
from mysr.core.functions import register_function

@register_function("stretchedExpo", 2, "exp(-(lambda*t)**beta)", custom=True)
def stretched_expo(t, lam, beta):
    return np.exp(-(lam * np.asarray(t))**beta)

# Then use in THEORY, e.g.:
#   asymmetry 1
#   stretchedExpo 2 3
# A custom numerical two-state strong-collision model can use the
# same interface once verified against Phys. Rev. B 112, 024309.
'''


def activate_user_code(source: str) -> list[str]:
    """Run user-authored Python explicitly; this is NOT a sandbox.

    On failure restore the registry to its previous contents.
    """
    before = dict(REGISTRY)
    try:
        exec(compile(source, "<mysr-custom-model>", "exec"), {"__name__": "mysr_user_model"})
    except Exception:
        REGISTRY.clear()
        REGISTRY.update(before)
        raise
    return sorted(k for k in REGISTRY if k not in before or REGISTRY[k] != before[k])
