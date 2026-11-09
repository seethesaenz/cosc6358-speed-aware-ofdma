"""Plot the pilot comparison from results/summary.csv (error bars = std across seeds).
Usage: python scripts/plot_pilot.py"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from _paths import ROOT

LABELS = {"perfect_csi": "Perfect CSI\n(τ = 0)", "pedestrian_3kmh": "Pedestrian\n3 km/h, τ = 2 ms",
          "vehicle_120kmh": "Vehicle\n120 km/h, τ = 2 ms"}
ALLOC = {"static|0dB": ("Static, rate at estimate (0 dB)", "#bfbfbf"), "static|3dB": ("Static, 3 dB back-off", "#7f7f7f"),
         "sum_rate|0dB": ("Sum-rate, rate at estimate (0 dB)", "#9dc3e6"), "sum_rate|3dB": ("Sum-rate, 3 dB back-off", "#1f4e79")}


def main():
    rows = list(csv.DictReader(open(ROOT / "results" / "summary.csv")))
    conds = list(dict.fromkeys(r["condition"] for r in rows))
    allocs = list(dict.fromkeys(r["variant"] for r in rows))
    panels = [("delivered_sum_mbps", "Delivered sum rate (Mbps)"), ("outage_pct", "Subcarrier outage (%)"),
              ("worst_user_mbps", "Worst-user rate (Mbps)")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    x = np.arange(len(conds)); w = 0.2
    for ax, (m, title) in zip(axes, panels):
        for i, a in enumerate(allocs):
            mean = [float(next(r for r in rows if r["condition"] == c and r["variant"] == a)[m + "_mean"]) for c in conds]
            std = [float(next(r for r in rows if r["condition"] == c and r["variant"] == a)[m + "_std"]) for c in conds]
            ax.bar(x + (i - (len(allocs) - 1) / 2) * w, mean, w, yerr=std, capsize=3, label=ALLOC[a][0], color=ALLOC[a][1])
        ax.set_xticks(x); ax.set_xticklabels([LABELS.get(c, c) for c in conds], fontsize=8)
        ax.set_title(title, fontsize=10); ax.grid(axis="y", alpha=0.3)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, frameon=False)
    axes[2].text(0.72, 0.82, "Sum-rate: worst user ≈ 0\n(weakest user almost never gets a subcarrier)", transform=axes[2].transAxes,
                 ha="center", fontsize=8, color="#1f4e79")
    fig.suptitle("MP2 pilot (synthetic): K = 4 users, N = 64 subcarriers, B = 1 MHz, 1 W, 5 seeds × 200 drops; bars = mean, error = std across seeds", fontsize=9)
    fig.tight_layout(rect=[0, 0.07, 1, 0.94])
    out = ROOT / "results" / "fig_pilot_comparison.png"
    fig.savefig(out, dpi=200); print("saved", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
