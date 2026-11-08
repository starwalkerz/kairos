"""
KAIROS smoke test.

Verifies the mathematical specification (Lemma 1, Theorem 1, the residual
and amplitude definitions) and the end-to-end detector behavior on a
synthetic series with and without an anomaly. No server required.

Run: python test_smoke.py

Note on the synthetic series: the baseline peak is A * exp(-(mu_L - mu_T)^2 /
(2*(sigma_L^2 + sigma_T^2))). With the parameters used here — mu_L = -0.5*T,
mu_T = +0.5*T, sigma_L = sigma_T = 0.25*T — the exponent evaluates to -4, so
the baseline peak is A * exp(-4) ≈ 0.0183 * A. Setting A = 55.0 yields a
baseline peak of approximately 1.0, matching the noise scale (0.02) and the
spike scale (0.3) such that the with-anomaly amplitude exceeds the 0.05
threshold and the without-anomaly amplitude is below it.
"""

from __future__ import annotations

import numpy as np

from core import (
    GmGCycle,
    from_preset,
    product_parameters,
    envelope,
    bayes_threshold,
    detection_window,
    cycle_local_time,
    baseline,
    residual,
    combined_window_mask,
    anomaly_amplitude,
)
from detect import detect


def _approx(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def test_lemma_1():
    """envelope(c, t) == A' * exp(-(t-mu)^2/(2 sigma^2)) for random cycles."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        c = GmGCycle(
            name="t",
            period=100.0,
            phase0=rng.uniform(-10, 10),
            mu_L=rng.uniform(-20, -5),
            sigma_L=rng.uniform(2, 10),
            mu_T=rng.uniform(5, 20),
            sigma_T=rng.uniform(2, 10),
            amplitude=rng.uniform(0.5, 2.0),
        )
        mu, sigma, A_prime = product_parameters(c)
        t = np.linspace(-50, 50, 1001)
        direct = envelope(c, t)
        via_product = A_prime * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2))
        max_diff = float(np.max(np.abs(direct - via_product)))
        assert max_diff < 1e-12, f"Lemma 1 violated: max diff = {max_diff}"
    print("[PASS] Lemma 1: product Gaussian matches envelope (max abs diff < 1e-12)")


def test_theorem_1_symmetric():
    """Symmetric cycle: Bayes threshold at midpoint."""
    c = GmGCycle(
        name="sym", period=90.0, phase0=0.0,
        mu_L=-45.0, sigma_L=22.5,
        mu_T=+45.0, sigma_T=22.5,
        amplitude=1.0,
    )
    x = bayes_threshold(c)
    assert _approx(x, 0.0, 1e-9), f"Expected midpoint 0, got {x}"
    print("[PASS] Theorem 1: Bayes threshold at midpoint for symmetric cycle")


def test_theorem_1_likelihoods():
    """At the threshold, the two likelihoods are equal."""
    from scipy import stats
    c = GmGCycle(
        name="asym", period=90.0, phase0=0.0,
        mu_L=-30.0, sigma_L=10.0,
        mu_T=+40.0, sigma_T=20.0,
        amplitude=1.0,
    )
    x = bayes_threshold(c)
    f_L = stats.norm.pdf((x - c.mu_L) / c.sigma_L) / c.sigma_L
    f_T = stats.norm.pdf((x - c.mu_T) / c.sigma_T) / c.sigma_T
    rel_err = abs(f_L - f_T) / max(f_L, f_T)
    assert rel_err < 1e-10, f"Likelihoods unequal at threshold: rel_err={rel_err}"
    print("[PASS] Theorem 1: likelihoods equal at threshold (rel err < 1e-10)")


def test_baseline_periodic():
    """Baseline is periodic: b(t) = b(t + k*T)."""
    c = GmGCycle(
        name="per", period=100.0, phase0=0.0,
        mu_L=-25.0, sigma_L=15.0,
        mu_T=+25.0, sigma_T=15.0,
        amplitude=1.0,
    )
    t = np.array([10.0, 23.7, 50.0, 78.4])
    b0 = baseline(t, [c])
    b1 = baseline(t + c.period, [c])
    b2 = baseline(t + 2 * c.period, [c])
    assert np.max(np.abs(b0 - b1)) < 1e-9
    assert np.max(np.abs(b0 - b2)) < 1e-9
    print("[PASS] Baseline periodic: b(t) = b(t + kT) for k=1,2")


def test_window_mask_cycle_local():
    """Window mask wraps across multiple cycles."""
    c = GmGCycle(
        name="wm", period=90.0, phase0=0.0,
        mu_L=-45.0, sigma_L=22.5,
        mu_T=+45.0, sigma_T=22.5,
        amplitude=1.0,
    )
    t = np.array([0.0, 20.0, 44.9, 45.1, 90.0, 110.0, 134.9, 135.1])
    mask = combined_window_mask(t, [c], kappa=2.0)
    _, w_lo, w_hi = detection_window(c, kappa=2.0)
    local = cycle_local_time(t, c)
    expected = (local >= w_lo) & (local <= w_hi)
    assert np.array_equal(mask, expected), "Window mask not cycle-local"
    assert mask[0], "t=0 should be inside window"
    assert mask[4], "t=90 should be inside window"
    assert not mask[2], "t=44.9 should be outside window"
    print("[PASS] Window mask: cycle-local wrapping across multiple cycles")


def _make_series(T, n, anomaly: bool):
    """Synthetic series with or without an anomaly.

    The amplitude is 55.0 so that the baseline peak — A * exp(-4) — is
    approximately 1.0, larger than the noise (0.02) and larger than the
    spike (0.3), which makes the residual amplitude interpretable as a
    fraction of total series energy.
    """
    t = np.linspace(0, 3 * T, n)
    c = GmGCycle(
        name="syn", period=T, phase0=0.0,
        mu_L=-0.5 * T, sigma_L=0.25 * T,
        mu_T=+0.5 * T, sigma_T=0.25 * T,
        amplitude=55.0,
    )
    b = baseline(t, [c])
    noise = 0.02 * np.sin(7 * 2 * np.pi * t / T)
    if anomaly:
        spike = np.where((t > 105) & (t < 115), 0.3, 0.0)
    else:
        spike = np.zeros_like(t)
    x = b + noise + spike
    return x, t, c


def test_detector_with_anomaly():
    x, t, c = _make_series(T=90.0, n=400, anomaly=True)
    report = detect(x, t, [c], threshold=0.05, quantum=False)
    assert report.flagged, (
        f"Expected flagged; got classical_amplitude={report.classical_amplitude}"
    )
    print(f"[PASS] Detector: anomaly flagged (amplitude={report.classical_amplitude:.4f})")


def test_detector_without_anomaly():
    x, t, c = _make_series(T=90.0, n=400, anomaly=False)
    report = detect(x, t, [c], threshold=0.05, quantum=False)
    assert not report.flagged, (
        f"Unexpected flag; got classical_amplitude={report.classical_amplitude}"
    )
    print(f"[PASS] Detector: anomaly not flagged (amplitude={report.classical_amplitude:.4f})")


def main() -> int:
    test_lemma_1()
    test_theorem_1_symmetric()
    test_theorem_1_likelihoods()
    test_baseline_periodic()
    test_window_mask_cycle_local()
    test_detector_with_anomaly()
    test_detector_without_anomaly()
    print("All assertions passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())