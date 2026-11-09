@echo off
REM Reproduce every MP2 number and figure on Windows (run from the project folder).
python scripts\run_pilot.py --config config\pilot.json || exit /b 1
python scripts\summarize.py || exit /b 1
python scripts\plot_pilot.py || exit /b 1
python scripts\check_correlation.py || exit /b 1
python scripts\verification_table.py || exit /b 1
python -m pytest -v tests > evidence\pytest_output.txt
type evidence\pytest_output.txt
