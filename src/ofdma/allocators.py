"""Baseline allocators. Each returns (assign, power):
  assign[n] = user index that gets subcarrier n, power[n] = watts on subcarrier n.
Allocators only see the ESTIMATED gains (possibly outdated)."""
import numpy as np


def water_filling(cnr, p_total, iters=200):
    """Water-filling over parallel channels.

    cnr: channel-to-noise ratio per channel (gain / noise power), linear.
    Returns p_n = max(0, mu - 1/cnr_n) with sum p_n = p_total; mu found by bisection.
    """
    cnr = np.asarray(cnr, dtype=float)
    inv = 1.0 / cnr                      # the "floor" of each channel, in watts
    lo, hi = inv.min(), inv.min() + p_total   # at hi the best channel alone takes p_total
    for _ in range(iters):
        mu = 0.5 * (lo + hi)
        if np.maximum(0.0, mu - inv).sum() > p_total:
            hi = mu
        else:
            lo = mu
    p = np.maximum(0.0, 0.5 * (lo + hi) - inv)
    return p * (p_total / p.sum())       # remove the last bisection rounding error


def static_allocate(gain_est, noise_w, p_total):
    """Static baseline: subcarrier n goes to user n mod K, equal power. Ignores the channel."""
    n_users, n_sub = gain_est.shape
    assign = np.arange(n_sub) % n_users
    power = np.full(n_sub, p_total / n_sub)
    return assign, power


def sum_rate_allocate(gain_est, noise_w, p_total):
    """Sum-rate baseline (inspired by Jang & Lee 2003, own implementation):
    each subcarrier to the user with the largest estimated gain, then water-filling."""
    n_sub = gain_est.shape[1]
    assign = np.argmax(gain_est, axis=0)
    cnr = gain_est[assign, np.arange(n_sub)] / noise_w
    return assign, water_filling(cnr, p_total)


ALLOCATORS = {"static": static_allocate, "sum_rate": sum_rate_allocate}
