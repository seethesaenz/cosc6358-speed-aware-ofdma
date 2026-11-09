"""Rates, outage and fairness. Rates are chosen from the estimate and delivered on the true channel."""
import numpy as np

TOL = 1e-9


def link_rates(assign, power, gain_est, gain_true, noise_w, subcarrier_bw_hz, margin_db=0.0):
    """Per-subcarrier planned rate, delivered rate (bit/s) and outage flag.

    planned = df * log2(1 + p * g_est / noise / 10^(margin_db/10))   (rate picked from the estimate,
              optionally backed off by a fixed SNR margin)
    capacity = df * log2(1 + p * g_true / noise) (what the true channel supports)
    delivered = planned if planned <= capacity, else 0 (outage).
    """
    n = np.arange(len(assign))
    g_est = gain_est[assign, n]
    g_true = gain_true[assign, n]
    planned = subcarrier_bw_hz * np.log2(1 + power * g_est / noise_w / 10 ** (margin_db / 10))
    capacity = subcarrier_bw_hz * np.log2(1 + power * g_true / noise_w)
    outage = (power > 0) & (planned > capacity * (1 + TOL) + TOL)
    delivered = np.where(outage, 0.0, planned)
    return planned, delivered, outage


def per_user_sum(assign, values, n_users):
    """Sum a per-subcarrier quantity for each user."""
    return np.bincount(assign, weights=np.asarray(values, dtype=float), minlength=n_users)


def jain_index(rates):
    """Jain's fairness index (sum r)^2 / (K * sum r^2); 1 = equal, 1/K = one user gets all."""
    r = np.asarray(rates, dtype=float)
    denom = len(r) * np.sum(r ** 2)
    return float(np.sum(r) ** 2 / denom) if denom > 0 else float("nan")
