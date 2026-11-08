"""Configurable exploratory Fourier transforms for time-domain μSR spectra."""
from __future__ import annotations

import numpy as np


GAMMA_MU_MHZ_PER_GAUSS = 0.0135538817
WINDOWS = ("None", "Hann", "Hamming", "Blackman")
OUTPUTS = ("Magnitude", "Power", "Real", "Imaginary")
UNITS = ("MHz", "Gauss", "Tesla")


def _window(name: str, size: int) -> np.ndarray:
    if name == "None":
        return np.ones(size)
    if name == "Hann":
        return np.hanning(size)
    if name == "Hamming":
        return np.hamming(size)
    if name == "Blackman":
        return np.blackman(size)
    raise ValueError(f"Unknown Fourier window {name!r}")


def fourier_transform(
    time_us,
    values,
    *,
    window: str = "Hann",
    subtract_mean: bool = True,
    zero_padding: int = 2,
    phase_deg: float = 0.0,
    output: str = "Magnitude",
    normalize: bool = True,
    units: str = "MHz",
) -> tuple[np.ndarray, np.ndarray]:
    """Return a one-sided transform in MHz, Gauss, or Tesla."""
    time = np.asarray(time_us, dtype=float)
    signal = np.asarray(values, dtype=float)
    if time.ndim != 1 or signal.shape != time.shape or len(time) < 4:
        raise ValueError("Fourier transform needs at least four matching 1D samples")
    spacing = np.diff(time)
    dt = float(np.median(spacing))
    if dt <= 0 or not np.allclose(spacing, dt, rtol=1e-3, atol=1e-12):
        raise ValueError("Fourier transform requires uniformly spaced time samples")
    if not 1 <= int(zero_padding) <= 64:
        raise ValueError("Fourier zero-padding factor must be between 1 and 64")
    if output not in OUTPUTS:
        raise ValueError(f"Unknown Fourier output {output!r}")
    if units not in UNITS:
        raise ValueError(f"Unknown Fourier units {units!r}")

    prepared = signal - np.mean(signal) if subtract_mean else signal.copy()
    weights = _window(window, len(prepared))
    base_size = 1 << int(np.ceil(np.log2(len(prepared))))
    transform = np.fft.rfft(prepared * weights, n=base_size * int(zero_padding))
    if normalize:
        scale = float(np.sum(weights))
        if scale > 0:
            transform = 2.0 * transform / scale
    transform *= np.exp(1j * np.deg2rad(float(phase_deg)))

    if output == "Magnitude":
        result = np.abs(transform)
    elif output == "Power":
        result = np.abs(transform) ** 2
    elif output == "Real":
        result = transform.real
    else:
        result = transform.imag

    frequency = np.fft.rfftfreq(base_size * int(zero_padding), dt)
    if units == "Gauss":
        frequency = frequency / GAMMA_MU_MHZ_PER_GAUSS
    elif units == "Tesla":
        frequency = frequency / (GAMMA_MU_MHZ_PER_GAUSS * 1e4)
    return frequency, np.asarray(result, dtype=float)
