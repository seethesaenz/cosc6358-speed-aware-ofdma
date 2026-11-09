"""Check the channel-aging model: empirical correlation between estimated and true subcarrier
gains H should match rho = J0(2*pi*f_D*tau). Writes results/correlation_check.csv.
Usage: python scripts/check_correlation.py"""
import csv
import numpy as np
from _paths import ROOT
from ofdma.channel import aging_correlation, age_taps, draw_taps, freq_response, pdp_weights

FC, N, TAU = 3.5e9, 64, 2e-3


def main():
    rng = np.random.default_rng(2026)
    pdp = pdp_weights(6, 2.0)
    out = []
    for v in [0, 3, 30, 60, 120]:
        rho = aging_correlation(v, TAU, FC)
        taps = draw_taps(rng, 20000, pdp)
        new = age_taps(rng, taps, rho, pdp)
        H0, H1 = freq_response(taps, N), freq_response(new, N)
        emp = float(np.real(np.mean(H1 * np.conj(H0))) / np.mean(np.abs(H0) ** 2))
        out.append({"speed_kmh": v, "tau_ms": TAU * 1e3, "rho_theory": round(rho, 4), "rho_empirical": round(emp, 4),
                    "abs_error": round(abs(emp - rho), 4)})
        print(out[-1])
    with open(ROOT / "results" / "correlation_check.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)


if __name__ == "__main__":
    main()
