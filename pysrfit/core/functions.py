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
    composite: bool = False

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


@register_function(
    "generExpo", 2,
    "Stretched exponential: exp(-(λt)^β); rate [μs⁻¹], exponent β",
)
def stretched_expo(t, lam, beta):
    """Musrfit's generalized (stretched) exponential relaxation."""
    times = np.asarray(t, dtype=float)
    base = float(lam) * times
    exponent = float(beta)
    result = np.zeros_like(times)
    valid = base >= 0.0
    if exponent.is_integer():
        valid = np.ones_like(base, dtype=bool)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result[valid] = np.exp(-np.power(base[valid], exponent))
    return result


@register_function("simpleGss", 1, "Gaussian: exp(-σ²t²/2)")
def simple_gss(t, sigma):
    return np.exp(-0.5 * (sigma * np.asarray(t)) ** 2)


def _bessel_j0(values):
    """Vectorized J0 approximation without requiring SciPy."""
    values = np.asarray(values, dtype=float)
    absolute = np.abs(values)
    result = np.empty_like(absolute)
    near = absolute < 8.0
    y = absolute[near] ** 2
    numerator = (
        57568490574.0
        + y * (-13362590354.0
        + y * (651619640.7
        + y * (-11214424.18
        + y * (77392.33017
        + y * -184.9052456))))
    )
    denominator = (
        57568490411.0
        + y * (1029532985.0
        + y * (9494680.718
        + y * (59272.64853
        + y * (267.8532712 + y))))
    )
    result[near] = numerator / denominator
    far = ~near
    x = absolute[far]
    z = 8.0 / x
    y = z**2
    phase = x - np.pi / 4.0
    correction_cos = (
        1.0
        + y * (-0.001098628627
        + y * (0.00002734510407
        + y * (-0.000002073370639
        + y * 0.0000002093887211)))
    )
    correction_sin = (
        -0.01562499995
        + y * (0.0001430488765
        + y * (-0.000006911147651
        + y * (0.0000007621095161
        - y * 0.0000000934945152)))
    )
    result[far] = np.sqrt(2.0 / (np.pi * x)) * (
        np.cos(phase) * correction_cos - z * np.sin(phase) * correction_sin)
    result[absolute == 0.0] = 1.0
    return result


@register_function(
    "Bessel", 2,
    "Bessel J₀(2πνt + phase); phase [deg], frequency [MHz]",
)
def bessel(t, phase_deg, frequency_mhz):
    argument = (
        np.deg2rad(float(phase_deg))
        + 2.0 * np.pi * float(frequency_mhz) * np.asarray(t, dtype=float)
    )
    return _bessel_j0(argument)


@register_function("TFieldCos", 2, "cos(2πνt + phase[deg]); ν in MHz")
def tf_cos(t, phase_deg, freq_mhz):
    return np.cos(2 * np.pi * freq_mhz * np.asarray(t) + np.deg2rad(phase_deg))


@register_function("statGssKt", 1, "Static Gaussian Kubo–Toyabe")
def gaussian_kt(t, delta):
    q = (delta * np.asarray(t)) ** 2
    return (1 + 2 * (1 - q) * np.exp(-0.5 * q)) / 3


@register_function(
    "statExpKT", 1,
    "Static Lorentzian Kubo–Toyabe: damping [μs⁻¹]",
)
def lorentzian_kt(t, damping):
    times = np.asarray(t, dtype=float)
    at = abs(float(damping)) * times
    return (1.0 + 2.0 * (1.0 - at) * np.exp(-at)) / 3.0


def _spherical_bessel_0_and_1(values):
    values = np.asarray(values, dtype=float)
    small = np.abs(values) < 1e-3
    j0 = np.empty_like(values)
    j1 = np.empty_like(values)
    squared = values[small] ** 2
    j0[small] = 1.0 - squared / 6.0 + squared**2 / 120.0
    j1[small] = values[small] / 3.0 - values[small] * squared / 30.0
    regular = ~small
    x = values[regular]
    j0[regular] = np.sin(x) / x
    j1[regular] = (np.sin(x) - x * np.cos(x)) / x**2
    return j0, j1


def _lorentzian_lf_integral(time, frequency_mhz, damping):
    """Numerically integrate musrfit's non-analytic Lorentzian LF term."""
    maximum = float(np.max(time)) if len(time) else 0.0
    if maximum <= 0.0:
        return np.zeros_like(time)
    omega = 2.0 * np.pi * frequency_mhz
    step = 0.001
    if damping <= omega and 1.0 / frequency_mhz < 20.0:
        step = min(step, 1.0 / frequency_mhz / 20.0)
    elif damping > omega and damping <= 10.0 and damping / omega > 10.0:
        step = min(step, 0.00005)
    elif damping > omega and damping > 10.0:
        step = min(step, 1.0 / damping / 3000.0)
    count = int(np.ceil(maximum / step)) + 1
    if count > 1_000_000:
        raise ValueError(
            "Static Lorentzian LF KT requires excessive numerical resolution")
    grid = np.arange(count, dtype=float) * step
    j0, _j1 = _spherical_bessel_0_and_1(omega * grid)
    prefactor = damping * (1.0 + (damping / omega) ** 2)
    integrand = prefactor * np.exp(-damping * grid) * j0
    integral = np.zeros_like(grid)
    integral[1:] = np.cumsum(
        0.5 * step * (integrand[:-1] + integrand[1:]))
    return np.interp(time, grid, integral)


@register_function(
    "statExpKTLF", 2,
    "Static Lorentzian LF Kubo–Toyabe: frequency [MHz], damping [μs⁻¹]",
)
def lorentzian_kt_lf(t, frequency_mhz, damping):
    """Static Lorentzian Kubo–Toyabe in longitudinal applied field."""
    times = np.asarray(t, dtype=float)
    shape = times.shape
    flat = times.reshape(-1)
    frequency = abs(float(frequency_mhz))
    rate = abs(float(damping))
    result = np.ones_like(flat)
    positive = flat >= 0.0
    selected = flat[positive]
    if not len(selected) or rate < 1e-15:
        return result.reshape(shape)
    if frequency < 0.02 or rate / frequency > 159.1549:
        result[positive] = lorentzian_kt(selected, rate)
        return result.reshape(shape)
    omega = 2.0 * np.pi * frequency
    argument = omega * selected
    j0, j1 = _spherical_bessel_0_and_1(argument)
    exponential = np.exp(-rate * selected)
    ratio = rate / omega
    integral = _lorentzian_lf_integral(selected, frequency, rate)
    result[positive] = (
        1.0
        - ratio * j1 * exponential
        - ratio**2 * (j0 * exponential - 1.0)
        - integral
    )
    return result.reshape(shape)


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


def _bmw_dynamic_lorentzian_lf(time, frequency_mhz, damping, hopping_rate):
    """BMW parametrization used by musrfit in the fast/high-field limit."""
    omega = 2.0 * np.pi * frequency_mhz
    c = (1.15331, 1.64826, -0.71763, 3.0, 0.386683, -5.01876, 2.41854)
    d = (2.44056, 2.92063, 1.69581, 0.667277)
    omega2 = omega * omega
    omega3 = omega2 * omega
    omega4 = omega3 * omega
    hopping2 = hopping_rate * hopping_rate
    hopping3 = hopping2 * hopping_rate
    hopping4 = hopping3 * hopping_rate
    denominator = (
        omega4
        + d[0] * omega3 * hopping_rate
        + d[1] * omega2 * hopping2
        + d[2] * omega * hopping3
        + d[3] * hopping4
    )
    if denominator <= 0.0:
        return np.ones_like(time)
    b1 = (
        c[0] * omega3
        + c[1] * omega2 * hopping_rate
        + c[2] * omega * hopping2
    ) / denominator
    b2 = (
        c[3] * omega3
        + c[4] * omega2 * hopping_rate
        + c[5] * omega * hopping2
        + c[6] * hopping3
    ) / denominator
    j0, j1 = _spherical_bessel_0_and_1(omega * time)
    collision = np.exp(-hopping_rate * time)
    gamma_t = -4.0 / 3.0 * damping * (
        b1 * (1.0 - j0 * collision)
        + b2 * j1 * collision
        + (1.0 - b2 * omega / 3.0 - b1 * hopping_rate) * time
    )
    return np.exp(gamma_t)


@register_function(
    "dynExpKTLF", 3,
    "Dynamic Lorentzian LF Kubo–Toyabe: frequency [MHz], damping and hopping rate [μs⁻¹]",
)
def dynamic_lorentzian_kt_lf(t, frequency_mhz, damping, hopping_rate):
    """Strong-collision dynamic Lorentzian LF Kubo–Toyabe."""
    times = np.asarray(t, dtype=float)
    shape = times.shape
    flat = times.reshape(-1)
    frequency = abs(float(frequency_mhz))
    rate = abs(float(damping))
    hopping = abs(float(hopping_rate))
    result = np.ones_like(flat)
    positive = flat >= 0.0
    selected = flat[positive]
    if not len(selected) or rate < 1e-15:
        return result.reshape(shape)
    if hopping < 1e-15:
        result[positive] = lorentzian_kt_lf(selected, frequency, rate)
        return result.reshape(shape)

    omega = 2.0 * np.pi * frequency
    if hopping > 5.0 * rate or omega >= 30.0 * rate:
        result[positive] = _bmw_dynamic_lorentzian_lf(
            selected, frequency, rate, hopping)
        return result.reshape(shape)

    maximum = max(20.0, float(np.max(selected)))
    count = int(16.0 * maximum * frequency)
    if rate > 0.1:
        characteristic_time = 2.0 / rate
        count = max(count, int(25.0 * maximum / characteristic_time))
    count = max(count, 300)
    if count > 20_000:
        raise ValueError(
            "Dynamic Lorentzian LF KT requires excessive numerical resolution")
    step = maximum / count
    grid = np.arange(count, dtype=float) * step
    no_collision = (
        lorentzian_kt_lf(grid, frequency, rate) * np.exp(-hopping * grid)
    )
    dynamic = np.empty_like(no_collision)
    dynamic[0] = no_collision[0]
    prefactor = step * hopping
    denominator = 1.0 - 0.5 * prefactor * no_collision[0]
    if denominator <= 0.0:
        raise ValueError("Dynamic Lorentzian LF KT integration became unstable")
    for index in range(1, count):
        total = no_collision[index]
        total += 0.5 * prefactor * no_collision[index] * dynamic[0]
        if index > 1:
            total += prefactor * np.dot(
                no_collision[index - 1:0:-1], dynamic[1:index])
        dynamic[index] = total / denominator
    result[positive] = np.interp(selected, grid, dynamic)
    return result.reshape(shape)


def two_state_collision(t, rate1, rate2, weight, static1, static2):
    """Generalized two-state strong-collision polarization.

    ``static1`` and ``static2`` are callables which evaluate the two static
    polarization functions.  The model solves the coupled renewal equations
    directly, including asymmetric state-1 -> state-2 and state-2 -> state-1
    rates.  This is Eq. (10) of Phys. Rev. B 112, 024309 (2025), evaluated in
    the time domain without explicitly summing individual collision orders.
    """
    times = np.asarray(t, dtype=float)
    shape = times.shape
    flat = times.reshape(-1)
    first_rate = abs(float(rate1))
    second_rate = abs(float(rate2))
    first_weight = float(weight)

    # The dynamic construction is causal and is only needed at t >= 0.  For
    # negative display times retain the weighted static polarization.
    first_at_times = np.asarray(static1(flat), dtype=float)
    second_at_times = np.asarray(static2(flat), dtype=float)
    if first_at_times.shape != flat.shape or second_at_times.shape != flat.shape:
        raise ValueError("Two-state static functions must preserve the time shape")
    result = first_weight * first_at_times + (1.0 - first_weight) * second_at_times
    positive = flat >= 0.0
    selected = flat[positive]
    if not len(selected):
        return result.reshape(shape)

    maximum = float(np.max(selected))
    if maximum <= 0.0 or max(first_rate, second_rate) < 1e-15:
        return result.reshape(shape)

    # Resolve the measured sampling at least twice, and use a finer default
    # grid for packed data.  The rate-dependent limit also keeps the renewal
    # integration accurate when collisions are fast.
    unique_times = np.unique(selected)
    spacings = np.diff(unique_times)
    spacings = spacings[spacings > 1e-12]
    measured_step = float(np.median(spacings)) if len(spacings) else maximum
    target_step = min(0.005, 0.5 * measured_step)
    fastest_rate = max(first_rate, second_rate)
    if fastest_rate > 0.0:
        target_step = min(target_step, 0.25 / fastest_rate)
    target_step = max(target_step, np.finfo(float).eps * max(1.0, maximum))
    count = int(np.ceil(maximum / target_step)) + 1
    if count > 8_000:
        raise ValueError(
            "Two-state collision model requires more than 8000 integration "
            "points; shorten the time range or reduce the switching rate"
        )
    grid = np.linspace(0.0, maximum, max(count, 2))
    step = float(grid[1] - grid[0])

    static_first = np.asarray(static1(grid), dtype=float)
    static_second = np.asarray(static2(grid), dtype=float)
    if static_first.shape != grid.shape or static_second.shape != grid.shape:
        raise ValueError("Two-state static functions must preserve the time shape")
    if not np.all(np.isfinite(static_first)) or not np.all(np.isfinite(static_second)):
        raise ValueError("Two-state static function produced nonfinite values")

    free_first = static_first * np.exp(-first_rate * grid)
    free_second = static_second * np.exp(-second_rate * grid)
    kernel_first = first_rate * free_first
    kernel_second = second_rate * free_second
    dynamic_first = np.empty_like(grid)
    dynamic_second = np.empty_like(grid)
    dynamic_first[0] = free_first[0]
    dynamic_second[0] = free_second[0]

    endpoint_first = 0.5 * step * kernel_first[0]
    endpoint_second = 0.5 * step * kernel_second[0]
    denominator = 1.0 - endpoint_first * endpoint_second
    if abs(denominator) < 1e-12:
        raise ValueError("Two-state collision integration became unstable")

    for index in range(1, len(grid)):
        first_rhs = free_first[index]
        second_rhs = free_second[index]
        first_rhs += 0.5 * step * kernel_first[index] * dynamic_second[0]
        second_rhs += 0.5 * step * kernel_second[index] * dynamic_first[0]
        if index > 1:
            first_rhs += step * np.dot(
                kernel_first[1:index], dynamic_second[index - 1:0:-1]
            )
            second_rhs += step * np.dot(
                kernel_second[1:index], dynamic_first[index - 1:0:-1]
            )
        dynamic_first[index] = (
            first_rhs + endpoint_first * second_rhs
        ) / denominator
        dynamic_second[index] = (
            second_rhs + endpoint_second * first_rhs
        ) / denominator

    dynamic = first_weight * dynamic_first + (1.0 - first_weight) * dynamic_second
    result[positive] = np.interp(selected, grid, dynamic)
    if not np.all(np.isfinite(result)):
        raise ValueError("Two-state collision model produced nonfinite values")
    return result.reshape(shape)


# This entry is parsed specially by theory.evaluate_theory because its two
# static functions are themselves registered THEORY expressions.
REGISTRY["twostate"] = FunctionSpec(
    "twoState",
    3,
    two_state_collision,
    "Generalized two-state collision model: nu1, nu2, starting weight g, "
    "followed by two static functions separated by '|'; constrain 0 <= g <= 1",
    composite=True,
)


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
