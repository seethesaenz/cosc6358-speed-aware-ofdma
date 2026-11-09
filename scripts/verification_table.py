"""Print expected (hand-computed) vs observed values for the report's verification table.
Writes results/verification_table.csv. Usage: python scripts/verification_table.py"""
import csv
import numpy as np
from _paths import ROOT
from ofdma.allocators import static_allocate, sum_rate_allocate, water_filling
from ofdma.channel import aging_correlation, age_taps, draw_taps, pdp_weights
from ofdma.metrics import jain_index, link_rates


def fmt(x):
    return "[" + ", ".join(f"{v:.4f}" for v in np.atleast_1d(x)) + "]"


def main():
    rows = []

    def add(tid, case, expected, observed, ok, why):
        rows.append({"id": tid, "case": case, "expected": expected, "observed": observed,
                    "result": "PASS" if ok else "FAIL", "explanation": why})

    p = water_filling([10.0, 2.0], 1.0)
    add("T1", "Water-filling, CNR [10, 2], P = 1 W", "[0.7000, 0.3000] W", fmt(p) + " W",
        np.allclose(p, [0.7, 0.3]), "mu = (1 + 0.1 + 0.5)/2 = 0.8; matches course example")
    p = water_filling([10.0, 0.1], 1.0)
    add("T2", "Boundary: very weak channel, CNR [10, 0.1]", "[1.0000, 0.0000] W", fmt(
        p) + " W", np.allclose(p, [1, 0]), "floor 10 W > water level, so channel 2 is switched off")
    a, pw = static_allocate(np.ones((3, 8)), 1.0, 1.0)
    add("T3", "Static split, K = 3, N = 8", "counts [3, 3, 2], 0.125 W each", f"counts {[int(c) for c in np.bincount(a)]}, {pw[0]:.3f} W each",
        list(np.bincount(a)) == [3, 3, 2] and np.allclose(pw, 0.125), "round-robin over subcarriers, equal power")
    a, pw = sum_rate_allocate(
        np.array([[4, 1, 3, 0.5], [2, 5, 1, 0.6]], float), 1.0, 1.0)
    add("T4", "Sum-rate on a 2 x 4 gain matrix", "assign [0, 1, 0, 1]; p [0.3444, 0.3944, 0.2611, 0]",
        f"assign {[int(x) for x in a]}; p {fmt(pw)}", list(a) == [
            0, 1, 0, 1] and np.allclose(pw, [0.344444, 0.394444, 0.261111, 0], atol=1e-6),
        "best user per subcarrier; mu = 0.5944 < 1.667, so subcarrier 4 is off")
    pl, de, out = link_rates(np.zeros(3, int), np.ones(3), np.array(
        [[7.0, 7, 7]]), np.array([[3.0, 15, 7]]), 1.0, 1.0)
    add("T5", "Outage rule: est. gain 7, true 3 / 15 / 7", "delivered [0, 3, 3]; outage [T, F, F]",
        f"delivered {fmt(de)}; outage {[bool(x) for x in out]}", np.allclose(
            de, [0, 3, 3]) and list(out) == [True, False, False],
        "planned 3 > capacity 2 fails; a better channel still delivers only the planned 3; a tie is not an outage")
    pl, de, out = link_rates(np.zeros(2, int), np.ones(2), np.array(
        [[7.0, 7]]), np.array([[3.0, 4]]), 1.0, 1.0, 3.0)
    add("T6", "3 dB back-off: est. 7, true 3 / 4", "planned 2.1726; delivered [0, 2.1726]", f"planned {pl[0]:.4f}; delivered {fmt(de)}",
        np.allclose(de, [0, 2.1726], atol=1e-4), "SNR 7/10^0.3 = 3.508 -> log2(4.508) = 2.1726")
    r = [aging_correlation(v, 2e-3, 3.5e9) for v in (0, 3, 120)]
    add("T7", "rho = J0(2 pi f_D tau), tau = 2 ms, 0 / 3 / 120 km/h", "[1.0000, 0.9963, -0.2138]", fmt(r),
        np.allclose(r, [1, 0.9963, -0.2138], atol=1e-4), "f_D = 0, 9.7, 388.9 Hz at 3.5 GHz")
    rng = np.random.default_rng(0)
    pdp = pdp_weights(6, 2.0)
    taps = draw_taps(rng, 4, pdp)
    d = float(np.max(np.abs(age_taps(rng, taps, 1.0, pdp) - taps)))
    add("T8", "Boundary: rho = 1 (no motion) leaves channel unchanged", "max |difference| = 0", f"max |difference| = {d:.1e}",
        d < 1e-12, "aging must be the identity when the user does not move")
    j = [jain_index([1, 1, 1, 1]), jain_index(
        [1, 0, 0, 0]), jain_index([2, 1])]
    add("T9", "Jain index: equal / one user / [2, 1]", "[1.0000, 0.2500, 0.9000]", fmt(j), np.allclose(j, [1, 0.25, 0.9]),
        "(sum r)^2 / (K sum r^2)")
    with open(ROOT / "results" / "verification_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for row in rows:
        print(row["id"], row["result"], "|",
              row["expected"], "|", row["observed"])


if __name__ == "__main__":
    main()
