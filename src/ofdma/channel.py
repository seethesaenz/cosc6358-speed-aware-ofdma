"""Frequency-selective multipath channel with Gauss-Markov aging.

Model (from the MP1 proposal):
  * L-tap Rayleigh channel per user, exponential power delay profile (PDP),
    total PDP power normalized to 1, so E|H[n]|^2 = 1 on every subcarrier.
  * Subcarrier response H[n] = FFT of the taps (N points).
  * Aging over a feedback delay tau:  h_new = rho * h_old + sqrt(1 - rho^2) * w,
    rho = J0(2*pi*f_D*tau), f_D = v * f_c / c  (Jakes correlation).
All quantities are synthetic (simulated), not measured.
"""
import numpy as np
from scipy.special import j0

SPEED_OF_LIGHT = 3e8  # m/s


def doppler_hz(speed_kmh, fc_hz):
    """Maximum Doppler shift in Hz for a speed in km/h and carrier in Hz."""
    return (speed_kmh / 3.6) * fc_hz / SPEED_OF_LIGHT


def aging_correlation(speed_kmh, tau_s, fc_hz):
    """Channel correlation rho = J0(2*pi*f_D*tau) between estimate and true channel."""
    return float(j0(2 * np.pi * doppler_hz(speed_kmh, fc_hz) * tau_s))


def pdp_weights(n_taps, decay_taps):
    """Exponential PDP: power of tap l proportional to exp(-l / decay_taps); sums to 1."""
    p = np.exp(-np.arange(n_taps) / decay_taps)
    return p / p.sum()


def draw_taps(rng, n_users, pdp):
    """Independent complex Gaussian taps, shape (n_users, n_taps), tap powers = pdp."""
    n_taps = len(pdp)
    w = (rng.standard_normal((n_users, n_taps)) + 1j * rng.standard_normal((n_users, n_taps))) / np.sqrt(2)
    return w * np.sqrt(pdp)


def age_taps(rng, taps, rho, pdp):
    """Advance the taps by one feedback delay with correlation rho (scalar or one value per user)."""
    rho = np.broadcast_to(np.asarray(rho, dtype=float), (taps.shape[0],)).reshape(-1, 1)
    w = draw_taps(rng, taps.shape[0], pdp)
    return rho * taps + np.sqrt(np.clip(1 - rho ** 2, 0.0, None)) * w


def freq_response(taps, n_sub):
    """Complex subcarrier gains H, shape (n_users, n_sub)."""
    return np.fft.fft(taps, n=n_sub, axis=1)


def power_gains(H, beta):
    """Power gains beta_k * |H_k[n]|^2, shape (n_users, n_sub)."""
    return np.asarray(beta).reshape(-1, 1) * np.abs(H) ** 2
