#!/usr/bin/env bash
# Reproduce every MP2 number and figure from scratch (run from the project root).
set -e
python scripts/run_pilot.py --config config/pilot.json   # raw CSVs + evidence/run_log.txt
python scripts/summarize.py                              # results/summary.csv, summary_by_seed.csv
python scripts/plot_pilot.py                             # results/fig_pilot_comparison.png
python scripts/check_correlation.py                      # results/correlation_check.csv
python scripts/verification_table.py                     # results/verification_table.csv
python -m pytest -v tests | tee evidence/pytest_output.txt
