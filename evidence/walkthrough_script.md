# MP2 video walkthrough (2 to 3 minutes)

**Jack, about 75 seconds: problem, channel model, pilot run**

1. (Show README) "Our project tests what happens to OFDMA allocation when channel feedback is a few milliseconds old and users move at different speeds. Everything here is synthetic, from our own simulator."
2. (Open src/ofdma/channel.py) "Each user has a six-tap Rayleigh channel; the FFT gives 64 subcarrier gains. Between feedback and transmission the channel ages with correlation rho equal to J0 of two pi f_D tau. At 3 km/h and 2 ms rho is 0.996; at 120 km/h it drops to minus 0.21, so the estimate is almost useless."
3. (Run `python scripts/run_pilot.py`) "This writes one raw CSV per condition and seed, with a timestamped log."
4. (Run `python scripts/check_correlation.py`) "The simulated correlation matches J0 within 0.004. I built allocators, tests and the reproduction checks."

**Abdullah, about 75 seconds: baseline, tests, result**

1. (Open src/ofdma/allocators.py) "Our main baseline is sum-rate: each subcarrier goes to the user with the best estimated gain, then water-filling. The static baseline splits subcarriers evenly with equal power."
2. (Run `python -m pytest -v tests`) "Sixteen tests pass. For example T1 reproduces the course example, 0.7 and 0.3 watts, and T5 checks that a rate above the true capacity counts as an outage."
3. (Run summarize and plot, show the figure) "With a 3 dB back-off, sum-rate delivers 5.15 Mbps with perfect CSI and almost the same at walking speed, but only 2.81 Mbps at 120 km/h, with 41% of subcarriers in outage. Without any back-off, even walking users lose about half the subcarriers."
4. "This is a five-seed pilot, so we are not claiming significance. Next we add the fair and speed-aware allocators. I built channel generator and aging model; rate, outage and fairness metrics; pilot runner, summary and plot scripts; and the report."
