import numpy as np
import pytest

from pysrfit.core.fourier import GAMMA_MU_MHZ_PER_GAUSS, fourier_transform


def test_fourier_transform_finds_frequency_and_converts_units():
    time = np.arange(0, 10, 0.01)
    signal = 0.2 * np.cos(2 * np.pi * 2.0 * time)
    frequency, magnitude = fourier_transform(
        time, signal, window="None", zero_padding=4, normalize=True)
    peak_mhz = frequency[np.argmax(magnitude[1:]) + 1]
    assert peak_mhz == pytest.approx(2.0, abs=0.01)

    field, field_magnitude = fourier_transform(
        time, signal, window="None", zero_padding=4,
        normalize=True, units="Gauss")
    peak_gauss = field[np.argmax(field_magnitude[1:]) + 1]
    assert peak_gauss == pytest.approx(2.0 / GAMMA_MU_MHZ_PER_GAUSS, rel=0.01)


def test_fourier_output_and_validation():
    time = np.arange(20) * 0.1
    signal = np.cos(2 * np.pi * 0.5 * time)
    frequency, real = fourier_transform(
        time, signal, output="Real", phase_deg=90, subtract_mean=False)
    assert frequency.shape == real.shape
    with pytest.raises(ValueError, match="uniformly spaced"):
        fourier_transform(np.array([0, 1, 2, 4.0]), np.ones(4))
