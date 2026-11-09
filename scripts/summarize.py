"""Summarize raw pilot CSVs into per-seed and across-seed tables.

Metrics per realization: delivered sum rate (Mbps), worst-user delivered rate (Mbps),
Jain's index of delivered user rates, outage fraction (outage / active subcarriers).
Per seed: mean over realizations. Final table: mean and std across seeds.
Usage: python scripts/summarize.py
"""
import csv, glob
from collections import defaultdict

import numpy as np
from _paths import ROOT
from ofdma.metrics import jain_index


def main():
    per_real = defaultdict(lambda: defaultdict(list))  # (cond, alloc, seed, real) -> lists
    meta = {}
    for path in sorted(glob.glob(str(ROOT / "data" / "raw" / "*.csv"))):
        with open(path) as f:
            for row in csv.DictReader(f):
                key = (row["condition"], f"{row['allocator']}|{float(row['margin_db']):g}dB", int(row["seed"]), int(row["realization"]))
                d = per_real[key]
                d["rate"].append(float(row["delivered_rate_bps"]))
                d["planned"].append(float(row["planned_rate_bps"]))
                d["active"].append(int(row["n_active_subcarriers"]))
                d["outage"].append(int(row["n_outage_subcarriers"]))
                meta[row["condition"]] = (row["speed_kmh"], row["tau_ms"], row["rho"])

    by_seed = defaultdict(lambda: defaultdict(list))
    for (cond, alloc, seed, _), d in per_real.items():
        r = np.array(d["rate"]) / 1e6
        s = by_seed[(cond, alloc, seed)]
        s["sum"].append(r.sum()); s["min"].append(r.min()); s["jain"].append(jain_index(r))
        s["planned"].append(np.sum(d["planned"]) / 1e6)
        s["outage"].append(np.sum(d["outage"]) / max(1, np.sum(d["active"])))

    seed_rows = []
    for (cond, alloc, seed), s in sorted(by_seed.items()):
        seed_rows.append({"condition": cond, "allocator": alloc, "seed": seed,
                          "planned_sum_mbps": np.mean(s["planned"]), "delivered_sum_mbps": np.mean(s["sum"]),
                          "worst_user_mbps": np.mean(s["min"]), "jain": np.nanmean(s["jain"]),
                          "outage_pct": 100 * np.mean(s["outage"])})
    res = ROOT / "results"; res.mkdir(exist_ok=True)
    with open(res / "summary_by_seed.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(seed_rows[0])); w.writeheader()
        for r in seed_rows: w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    metrics = ["planned_sum_mbps", "delivered_sum_mbps", "worst_user_mbps", "jain", "outage_pct"]
    order = [c for c in ["perfect_csi", "pedestrian_3kmh", "vehicle_120kmh"] if c in meta] + \
            sorted(c for c in meta if c not in ["perfect_csi", "pedestrian_3kmh", "vehicle_120kmh"])
    final = []
    for cond in order:
        for alloc in sorted({r["allocator"] for r in seed_rows}, key=lambda a: (a.split("|")[0] != "static", a)):
            rs = [r for r in seed_rows if r["condition"] == cond and r["allocator"] == alloc]
            row = {"condition": cond, "speed_kmh": meta[cond][0], "tau_ms": meta[cond][1], "rho": meta[cond][2],
                   "allocator": alloc.split("|")[0], "margin_db": alloc.split("|")[1].replace("dB", ""),
                   "variant": alloc, "n_seeds": len(rs)}
            for m in metrics:
                v = np.array([r[m] for r in rs])
                row[m + "_mean"] = round(float(v.mean()), 4); row[m + "_std"] = round(float(v.std(ddof=1)), 4)
            final.append(row)
    with open(res / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(final[0])); w.writeheader(); w.writerows(final)
    for r in final:
        print(f"{r['condition']:16s} {r['variant']:13s} rho={float(r['rho']):+.3f}  "
              f"delivered={r['delivered_sum_mbps_mean']:.3f}±{r['delivered_sum_mbps_std']:.3f} Mbps  "
              f"worst={r['worst_user_mbps_mean']:.3f}  jain={r['jain_mean']:.3f}  outage={r['outage_pct_mean']:.2f}%")


if __name__ == "__main__":
    main()
