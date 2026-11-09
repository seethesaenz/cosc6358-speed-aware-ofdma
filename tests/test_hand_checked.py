"""Hand-checked cases. Each test states the expected value worked out by hand in the comment.
IDs (T1, T2, ...) match the verification table in the MP2 report."""
import numpy as np
import pytest

from ofdma.allocators import static_allocate, sum_rate_allocate, water_filling
from ofdma.channel import aging_correlation, age_taps, doppler_hz, draw_taps, freq_response, pdp_weights
from ofdma.metrics import jain_index, link_rates


def test_T1_water_filling_course_example():
    # Module 1/4 example: N = 0.1, |h1|^2 = 1, |h2|^2 = 0.2 -> CNR = [10, 2], Pmax = 1.
    # Floors 1/CNR = 0.1 and 0.5; mu = (1 + 0.1 + 0.5)/2 = 0.8; p = [0.7, 0.3].
    p = water_filling([10.0, 2.0], 1.0)
    assert p == pytest.approx([0.7, 0.3], abs=1e-9)
    rates = np.log2(1 + np.array([10.0, 2.0]) * p)       # log2(8) = 3, log2(1.6) = 0.678
    assert rates == pytest.approx([3.0, 0.678072], abs=1e-6)


def test_T2_water_filling_boundary_weak_channel_switched_off():
    # CNR = [10, 0.1]: floors 0.1 and 10. Both active would need mu = (1 + 10.1)/2 = 5.55 < 10,
    # so channel 2 gets 0 and channel 1 takes all power: p = [1, 0].
    assert water_filling([10.0, 0.1], 1.0) == pytest.approx([1.0, 0.0], abs=1e-9)


def test_T3_static_split():
    # K = 3, N = 8 -> users 0,1,2,0,1,2,0,1 -> counts [3, 3, 2]; equal power 1/8 W each.
    assign, power = static_allocate(np.ones((3, 8)), 1.0, 1.0)
    assert list(assign) == [0, 1, 2, 0, 1, 2, 0, 1]
    assert list(np.bincount(assign)) == [3, 3, 2]
    assert power == pytest.approx(np.full(8, 0.125)) and power.sum() == pytest.approx(1.0)


def test_T4_sum_rate_assignment_and_power():
    # Gains (noise = 1): user0 = [4, 1, 3, 0.5], user1 = [2, 5, 1, 0.6]
    # Best user per subcarrier -> [0, 1, 0, 1], CNR = [4, 5, 3, 0.6], floors [0.25, 0.2, 0.333, 1.667].
    # With the first three active: mu = (1 + 0.25 + 0.2 + 1/3)/3 = 0.594444 < 1.667, so subcarrier 4 gets 0.
    # p = [0.344444, 0.394444, 0.261111, 0].
    g = np.array([[4, 1, 3, 0.5], [2, 5, 1, 0.6]], dtype=float)
    assign, power = sum_rate_allocate(g, 1.0, 1.0)
    assert list(assign) == [0, 1, 0, 1]
    assert power == pytest.approx([0.344444, 0.394444, 0.261111, 0.0], abs=1e-6)


def test_T5_outage_rule():
    # df = 1 Hz, noise = 1, p = 1. Estimate gain 7 -> planned log2(8) = 3 bit/s.
    # True gain 3 -> capacity log2(4) = 2 < 3 -> outage, delivered 0.
    # True gain 15 -> capacity 4 >= 3 -> delivered 3 (the planned rate, not 4).
    # Tie: true gain 7 -> capacity 3 = planned -> no outage.
    assign, power = np.array([0, 0, 0]), np.ones(3)
    g_est = np.array([[7.0, 7.0, 7.0]])
    g_true = np.array([[3.0, 15.0, 7.0]])
    planned, delivered, outage = link_rates(assign, power, g_est, g_true, 1.0, 1.0)
    assert planned == pytest.approx([3, 3, 3])
    assert delivered == pytest.approx([0, 3, 3])
    assert list(outage) == [True, False, False]


def test_T6_outage_with_3db_margin():
    # 3 dB back-off: SNR used = 7 / 10^0.3 = 3.5084 -> planned log2(4.5084) = 2.1726.
    # True gain 3 (capacity 2) -> still outage; true gain 4 (capacity log2 5 = 2.3219) -> delivered 2.1726.
    assign, power = np.array([0, 0]), np.ones(2)
    planned, delivered, outage = link_rates(assign, power, np.array([[7.0, 7.0]]), np.array([[3.0, 4.0]]), 1.0, 1.0, 3.0)
    assert planned == pytest.approx([2.1726, 2.1726], abs=1e-4)
    assert delivered == pytest.approx([0.0, 2.1726], abs=1e-4)
    assert list(outage) == [True, False]


def test_T7_doppler_and_correlation_values():
    # f_D = (120/3.6) * 3.5e9 / 3e8 = 388.9 Hz. rho = J0(2*pi*f_D*tau).
    assert doppler_hz(120, 3.5e9) == pytest.approx(388.889, abs=1e-3)
    assert aging_correlation(0, 2e-3, 3.5e9) == pytest.approx(1.0)          # no motion -> perfect
    assert aging_correlation(3, 2e-3, 3.5e9) == pytest.approx(0.9963, abs=1e-4)
    assert aging_correlation(120, 2e-3, 3.5e9) == pytest.approx(-0.2138, abs=1e-4)


def test_T8_aging_boundaries():
    # rho = 1 must return the channel unchanged (stationary user, no delay).
    rng = np.random.default_rng(0)
    pdp = pdp_weights(6, 2.0)
    taps = draw_taps(rng, 4, pdp)
    assert np.allclose(age_taps(rng, taps, 1.0, pdp), taps)


def test_T9_jain_index():
    # Equal rates -> 1; one user gets everything (K = 4) -> 1/4; [2, 1] -> 9 / (2 * 5) = 0.9.
    assert jain_index([1, 1, 1, 1]) == pytest.approx(1.0)
    assert jain_index([1, 0, 0, 0]) == pytest.approx(0.25)
    assert jain_index([2, 1]) == pytest.approx(0.9)


def test_T10_channel_power_normalization():
    # PDP sums to 1, so the average subcarrier power gain E|H|^2 should be 1 (statistical check).
    rng = np.random.default_rng(1)
    pdp = pdp_weights(6, 2.0)
    assert pdp.sum() == pytest.approx(1.0)
    H = freq_response(draw_taps(rng, 20000, pdp), 64)
    assert np.mean(np.abs(H) ** 2) == pytest.approx(1.0, abs=0.02)


def test_T11_aging_correlation_matches_j0():
    # Empirical corr(H_old, H_new) should match J0 for 30 km/h, tau = 2 ms (rho = 0.660).
    rng = np.random.default_rng(2)
    pdp = pdp_weights(6, 2.0)
    rho = aging_correlation(30, 2e-3, 3.5e9)
    old = draw_taps(rng, 20000, pdp)
    new = age_taps(rng, old, rho, pdp)
    H0, H1 = freq_response(old, 64), freq_response(new, 64)
    emp = np.real(np.mean(H1 * np.conj(H0))) / np.mean(np.abs(H0) ** 2)
    assert emp == pytest.approx(rho, abs=0.01)
