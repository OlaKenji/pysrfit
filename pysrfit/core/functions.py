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


def _gaussian_lf_integral(time, frequency_mhz, delta):
    """Numerically integrate the non-analytic term in static Gaussian LF KT."""
    maximum = float(np.max(time)) if len(time) else 0.0
    if maximum <= 0.0:
        return np.zeros_like(time)
    omega = 2.0 * np.pi * frequency_mhz
    step = 0.001
    if delta <= omega and 1.0 / frequency_mhz < 20.0:
        step = min(step, 1.0 / frequency_mhz / 20.0)
    elif delta > omega and delta <= 10.0 and delta / omega > 10.0:
        step = min(step, 0.00005)
    elif delta > omega and delta > 10.0:
        step = min(step, 1.0 / delta / 3000.0)
    count = int(np.ceil(maximum / step)) + 1
    if count > 1_000_000:
        raise ValueError("Static Gaussian LF KT requires excessive numerical resolution")
    grid = np.arange(count, dtype=float) * step
    prefactor = 2.0 * delta**4 / omega**3
    integrand = prefactor * np.exp(-0.5 * (delta * grid) ** 2) * np.sin(omega * grid)
    integral = np.zeros_like(grid)
    integral[1:] = np.cumsum(0.5 * step * (integrand[:-1] + integrand[1:]))
    return np.interp(time, grid, integral)


@register_function(
    "statGssKTLF",
    2,
    "Static Gaussian LF Kubo–Toyabe: frequency [MHz], damping [μs⁻¹]",
)
def gaussian_kt_lf(t, frequency_mhz, delta):
    """Static Gaussian Kubo–Toyabe in a longitudinal applied field."""
    times = np.asarray(t, dtype=float)
    shape = times.shape
    flat = times.reshape(-1)
    frequency = abs(float(frequency_mhz))
    width = abs(float(delta))
    result = np.ones_like(flat)
    positive = flat >= 0.0
    selected = flat[positive]
    if not len(selected) or width < 1e-15:
        return result.reshape(shape)
    if frequency < 0.02 or width / frequency > 79.5775:
        result[positive] = gaussian_kt(selected, width)
        return result.reshape(shape)
    omega = 2.0 * np.pi * frequency
    integral = _gaussian_lf_integral(selected, frequency, width)
    ratio = width / omega
    result[positive] = (
        1.0
        - 2.0 * ratio**2 * (
            1.0
            - np.exp(-0.5 * (width * selected) ** 2)
            * np.cos(omega * selected)
        )
        + integral
    )
    return result.reshape(shape)


def _keren_gaussian_lf(time, frequency_mhz, delta, hopping_rate):
    omega = 2.0 * np.pi * frequency_mhz
    omega2 = omega * omega
    hopping2 = hopping_rate * hopping_rate
    denominator = (omega2 + hopping2) ** 2
    gamma_t = 2.0 * delta**2 / denominator * (
        (omega2 + hopping2) * hopping_rate * time
        + (omega2 - hopping2) * (
            1.0 - np.exp(-hopping_rate * time) * np.cos(omega * time)
        )
        - 2.0 * hopping_rate * omega
        * np.exp(-hopping_rate * time) * np.sin(omega * time)
    )
    return np.exp(-gamma_t)


@register_function(
    "dynGssKTLF",
    3,
    "Dynamic Gaussian LF Kubo–Toyabe: frequency [MHz], damping and hopping rate [μs⁻¹]",
)
def dynamic_gaussian_kt_lf(t, frequency_mhz, delta, hopping_rate):
    """Strong-collision dynamic Gaussian LF Kubo–Toyabe.

    The slow/intermediate regime uses the Volterra renewal equation with
    trapezoidal integration.  As in musrfit, ν/Δ > 5 uses Keren's
    fast-fluctuation approximation.
    """
    times = np.asarray(t, dtype=float)
    shape = times.shape
    flat = times.reshape(-1)
    frequency = abs(float(frequency_mhz))
    width = abs(float(delta))
    hopping = abs(float(hopping_rate))
    result = np.ones_like(flat)
    positive = flat >= 0.0
    selected = flat[positive]
    if not len(selected) or width < 1e-6:
        return result.reshape(shape)
    if hopping < 1e-15:
        result[positive] = gaussian_kt_lf(selected, frequency, width)
        return result.reshape(shape)
    if hopping / width > 5.0:
        result[positive] = _keren_gaussian_lf(
            selected, frequency, width, hopping)
        return result.reshape(shape)

    maximum = max(20.0, float(np.max(selected)))
    count = int(16.0 * maximum * frequency)
    if width > 0.1:
        characteristic_time = np.sqrt(3.0) / width
        count = max(count, int(25.0 * maximum / characteristic_time))
    count = max(count, 300)
    if count > 20_000:
        raise ValueError("Dynamic Gaussian LF KT requires excessive numerical resolution")
    step = maximum / count
    grid = np.arange(count, dtype=float) * step
    no_collision = gaussian_kt_lf(grid, frequency, width) * np.exp(-hopping * grid)
    dynamic = np.empty_like(no_collision)
    dynamic[0] = no_collision[0]
    prefactor = step * hopping
    denominator = 1.0 - 0.5 * prefactor * no_collision[0]
    if denominator <= 0.0:
        raise ValueError("Dynamic Gaussian LF KT integration became unstable")
    for index in range(1, count):
        total = no_collision[index]
        total += 0.5 * prefactor * no_collision[index] * dynamic[0]
        if index > 1:
            total += prefactor * np.dot(
                no_collision[index - 1:0:-1], dynamic[1:index])
        dynamic[index] = total / denominator
    result[positive] = np.interp(selected, grid, dynamic)
    return result.reshape(shape)


ALIASES = {
    "a": "asymmetry",
    "se": "simplExpo",
    "sg": "simpleGss",
    "tf": "TFieldCos",
    "sgktlf": "statGssKTLF",
    "dgktlf": "dynGssKTLF",
}

CUSTOM_EXAMPLE = '''# This code runs only when you click "Activate code".
# You can write your own Python relaxation functions without compilation.
import numpy as np
from pysrfit.core.functions import register_function

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
        exec(compile(source, "<pysrfit-custom-model>", "exec"), {"__name__": "pysrfit_user_model"})
    except Exception:
        REGISTRY.clear()
        REGISTRY.update(before)
        raise
    return sorted(k for k in REGISTRY if k not in before or REGISTRY[k] != before[k])
