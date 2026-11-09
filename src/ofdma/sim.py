"""Pilot simulation: one condition (speed, feedback delay) and one seed -> raw rows."""
import numpy as np

from .allocators import ALLOCATORS
from .channel import aging_correlation, age_taps, draw_taps, freq_response, pdp_weights, power_gains
from .metrics import link_rates

FIELDS = ["condition", "seed", "realization", "speed_kmh", "tau_ms", "rho", "allocator", "margin_db", "user",
          "beta_db", "n_subcarriers", "power_w", "planned_rate_bps", "delivered_rate_bps",
          "n_active_subcarriers", "n_outage_subcarriers"]


def derived(cfg):
    """Quantities computed from the config."""
    beta_db = np.linspace(cfg["beta_db_range"][0], cfg["beta_db_range"][1], cfg["n_users"])
    df = cfg["bandwidth_hz"] / cfg["n_subcarriers"]
    noise_w = 10 ** (cfg["noise_psd_dbw_hz"] / 10) * df
    return beta_db, 10 ** (beta_db / 10), df, noise_w


def run_condition(cfg, cond, seed):
    """Return a list of raw rows (dicts). The same seed gives the same channel draws in every
    condition, so conditions differ only in speed/delay (common random numbers)."""
    rng = np.random.default_rng(seed)
    beta_db, beta, df, noise_w = derived(cfg)
    pdp = pdp_weights(cfg["n_taps"], cfg["pdp_decay_taps"])
    rho = aging_correlation(cond["speed_kmh"], cond["tau_ms"] / 1e3, cfg["fc_hz"])
    K, N, P = cfg["n_users"], cfg["n_subcarriers"], cfg["p_total_w"]
    rows = []
    for r in range(cfg["realizations_per_seed"]):
        taps_est = draw_taps(rng, K, pdp)                 # channel when feedback was measured
        taps_true = age_taps(rng, taps_est, rho, pdp)     # channel at transmission time
        g_est = power_gains(freq_response(taps_est, N), beta)
        g_true = power_gains(freq_response(taps_true, N), beta)
        for name in cfg["allocators"]:
            assign, power = ALLOCATORS[name](g_est, noise_w, P)
            active = power > 0
            for margin in cfg.get("rate_margins_db", [0.0]):
                planned, delivered, outage = link_rates(assign, power, g_est, g_true, noise_w, df, margin)
                for k in range(K):
                    mine = assign == k
                    rows.append({
                        "condition": cond["name"], "seed": seed, "realization": r,
                        "speed_kmh": cond["speed_kmh"], "tau_ms": cond["tau_ms"], "rho": round(rho, 6),
                        "allocator": name, "margin_db": margin, "user": k,
                        "beta_db": round(float(beta_db[k]), 3),
                        "n_subcarriers": int(mine.sum()), "power_w": float(power[mine].sum()),
                        "planned_rate_bps": float(planned[mine].sum()),
                        "delivered_rate_bps": float(delivered[mine].sum()),
                        "n_active_subcarriers": int((mine & active).sum()),
                        "n_outage_subcarriers": int((mine & outage).sum()),
                    })
    return rows
