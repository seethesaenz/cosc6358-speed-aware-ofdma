"""Properties that must hold on every random draw."""
import json
from pathlib import Path

import numpy as np
import pytest

from ofdma.allocators import ALLOCATORS
from ofdma.sim import derived, run_condition

CFG = json.loads((Path(__file__).resolve().parents[1] / "config" / "pilot.json").read_text())


@pytest.mark.parametrize("name", list(ALLOCATORS))
def test_power_budget_and_one_user_per_subcarrier(name):
    rng = np.random.default_rng(5)
    for _ in range(50):
        g = rng.exponential(size=(4, 64)) * np.array([[1], [0.3], [0.1], [0.01]])
        assign, power = ALLOCATORS[name](g, 1.5625e-4, 1.0)
        assert assign.shape == (64,) and set(np.unique(assign)) <= {0, 1, 2, 3}   # one user each
        assert np.all(power >= 0) and power.sum() == pytest.approx(1.0, abs=1e-9)


def test_perfect_csi_has_no_outage():
    cfg = dict(CFG, realizations_per_seed=20)
    rows = run_condition(cfg, {"name": "t", "speed_kmh": 0, "tau_ms": 0}, seed=1)
    assert sum(r["n_outage_subcarriers"] for r in rows) == 0


def test_common_random_numbers_and_reproducibility():
    # Same seed -> identical planned rates in two conditions (same estimate draws), and a rerun is identical.
    cfg = dict(CFG, realizations_per_seed=5)
    a = run_condition(cfg, {"name": "a", "speed_kmh": 3, "tau_ms": 2}, seed=3)
    b = run_condition(cfg, {"name": "b", "speed_kmh": 120, "tau_ms": 2}, seed=3)
    a2 = run_condition(cfg, {"name": "a", "speed_kmh": 3, "tau_ms": 2}, seed=3)
    assert [r["planned_rate_bps"] for r in a] == [r["planned_rate_bps"] for r in b]
    assert a == a2


def test_noise_per_subcarrier():
    # N0 = -80 dBW/Hz = 1e-8 W/Hz; df = 1 MHz / 64 = 15625 Hz -> noise = 1.5625e-4 W.
    _, _, df, noise = derived(CFG)
    assert df == pytest.approx(15625.0) and noise == pytest.approx(1.5625e-4)
