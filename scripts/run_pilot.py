"""Run the MP2 pilot and write raw CSV files, a run log and the environment record.

Usage (from the project root):  python scripts/run_pilot.py --config config/pilot.json
"""
import argparse, csv, hashlib, json, platform, sys, time
from datetime import datetime, timezone

from _paths import ROOT
import numpy, scipy
from ofdma.sim import FIELDS, run_condition


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/pilot.json")
    args = ap.parse_args()
    cfg_path = ROOT / args.config
    cfg_bytes = cfg_path.read_bytes()
    cfg = json.loads(cfg_bytes)
    cfg_hash = hashlib.sha256(cfg_bytes).hexdigest()[:12]

    raw_dir = ROOT / "data" / "raw"; raw_dir.mkdir(parents=True, exist_ok=True)
    ev_dir = ROOT / "evidence"; ev_dir.mkdir(exist_ok=True)

    (ev_dir / "environment.txt").write_text(
        f"python {sys.version.split()[0]}\nnumpy {numpy.__version__}\nscipy {scipy.__version__}\n"
        f"platform {platform.platform()}\n")

    log = open(ev_dir / "run_log.txt", "w")
    def say(msg):
        line = f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}  {msg}"
        print(line); log.write(line + "\n")

    say(f"START experiment={cfg['experiment']} config={args.config} sha256[:12]={cfg_hash} SYNTHETIC DATA")
    for cond in cfg["conditions"]:
        for seed in cfg["seeds"]:
            t0 = time.time()
            rows = run_condition(cfg, cond, seed)
            out = raw_dir / f"{cond['name']}_seed{seed}.csv"
            with open(out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
            say(f"wrote {out.relative_to(ROOT)} rows={len(rows)} rho={rows[0]['rho']} ({time.time()-t0:.2f}s)")
    say("DONE")
    log.close()


if __name__ == "__main__":
    main()
