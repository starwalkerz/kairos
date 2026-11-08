"""
core.py — cycle algebra, the Gaussian-modulated-Gaussian lemma, the Bayes
decision threshold, the cyclical baseline, the residual, and the anomaly
amplitude.

No project-internal imports. Depends only on numpy and scipy.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
from scipy import optimize, stats


SYNODIC_PRESETS: dict[str, float] = {
    "lunar_synodic": 29.5306,
    "mercury_sun":   115.88,
    "venus_sun":     583.92,
    "mars_sun":      779.94,
    "jupiter_sun":   398.88,
    "saturn_sun":    378.09,
    "uranus_sun":    369.66,
    "neptune_sun":   367.49,
}


def synodic_period(T1: float, T2: float) -> float:
    """Synodic (beat) period of two siderial periods: 1/T = |1/T1 - 1/T2|."""
    if T1 <= 0 or T2 <= 0:
        raise ValueError("periods must be positive")
    inv = abs(1.0 / T1 - 1.0 / T2)
    if inv == 0:
        raise ValueError("identical periods produce no synodic beat")
    return 1.0 / inv


@dataclass(frozen=True)
class GmGCycle:
    """A Gaussian-modulated Gaussian cycle.

    Attributes
    ----------
    name       : label
    period     : T, in the units of the time axis
    phase0     : phase offset at t = 0
    mu_L       : leading-Gaussian centre
    sigma_L    : leading-Gaussian width (> 0)
    mu_T       : trailing-Gaussian centre
    sigma_T    : trailing-Gaussian width (> 0)
    amplitude  : A, non-negative
    """
    name: str
    period: float
    phase0: float
    mu_L: float
    sigma_L: float
    mu_T: float
    sigma_T: float
    amplitude: float = 1.0

    def __post_init__(self) -> None:
        if self.period <= 0:
            raise ValueError("period must be positive")
        if self.sigma_L <= 0 or self.sigma_T <= 0:
            raise ValueError("sigma_L and sigma_T must be positive")
        if self.amplitude < 0:
            raise ValueError("amplitude must be non-negative")


def product_parameters(c: GmGCycle) -> tuple[float, float, float]:
    """Lemma 1: (mu, sigma, A') of g_C(t) as a single Gaussian."""
    precision = 1.0 / c.sigma_L**2 + 1.0 / c.sigma_T**2
    sigma = precision ** -0.5
    mu = sigma**2 * (c.mu_L / c.sigma_L**2 + c.mu_T / c.sigma_T**2)
    log_A = np.log(c.amplitude) if c.amplitude > 0 else -np.inf
    log_A -= (c.mu_L - c.mu_T) ** 2 / (2.0 * (c.sigma_L**2 + c.sigma_T**2))
    A_prime = float(np.exp(log_A)) if np.isfinite(log_A) else 0.0
    return float(mu), float(sigma), A_prime


def envelope(c: GmGCycle, t: np.ndarray) -> np.ndarray:
    """Evaluate g_C(t) directly from Definition 1."""
    return c.amplitude * np.exp(
        -((t - c.mu_L) ** 2) / (2.0 * c.sigma_L**2)
        -((t - c.mu_T) ** 2) / (2.0 * c.sigma_T**2)
    )


def _likelihood_difference(x: float, c: GmGCycle) -> float:
    left = stats.norm.pdf((x - c.mu_L) / c.sigma_L) / c.sigma_L
    right = stats.norm.pdf((x - c.mu_T) / c.sigma_T) / c.sigma_T
    return float(left - right)


def bayes_threshold(c: GmGCycle) -> float:
    """Theorem 1: unique Bayes decision threshold on [mu_L, mu_T].

    Returns the unique root of f_L(x) = f_T(x), where f_L, f_T are the
    signal and noise densities. When sigma_L = sigma_T, the root is the
    midpoint of mu_L and mu_T.
    """
    lo, hi = sorted((c.mu_L, c.mu_T))
    if hi - lo < 1e-12:
        return float(0.5 * (lo + hi))

    f_lo = _likelihood_difference(lo, c)
    f_hi = _likelihood_difference(hi, c)

    if f_lo == 0.0:
        return float(lo)
    if f_hi == 0.0:
        return float(hi)

    if np.sign(f_lo) == np.sign(f_hi):
        # Degenerate case: fall back to the product mean.
        mu, _, _ = product_parameters(c)
        return float(mu)

    return float(optimize.brentq(
        lambda x: _likelihood_difference(x, c),
        lo, hi, xtol=1e-12, rtol=1e-12, maxiter=200,
    ))


# Backward-compatible alias for the former name.
nash_detection_point = bayes_threshold


def detection_window(
    c: GmGCycle, kappa: float = 2.0
) -> tuple[float, float, float]:
    """Return (threshold, window_lo, window_hi) for the cycle."""
    x_star = bayes_threshold(c)
    _, sigma, _ = product_parameters(c)
    return x_star, x_star - kappa * sigma, x_star + kappa * sigma


def cycle_local_time(t: np.ndarray, c: GmGCycle) -> np.ndarray:
    """Map global t to cycle-local time in [-T/2, T/2)."""
    phase_fraction = (t - c.phase0) / c.period
    return ((phase_fraction + 0.5) % 1.0 - 0.5) * c.period


def baseline(t: np.ndarray, cycles: Sequence[GmGCycle]) -> np.ndarray:
    """Sum of phase-wrapped GmG envelopes over the cycle stack."""
    out = np.zeros_like(t, dtype=float)
    for c in cycles:
        wrapped_time = cycle_local_time(t, c)
        out += envelope(c, wrapped_time)
    return out


def residual(
    x: np.ndarray, t: np.ndarray, cycles: Sequence[GmGCycle]
) -> np.ndarray:
    return x - baseline(t, cycles)


def combined_window_mask(
    t: np.ndarray, cycles: Sequence[GmGCycle], kappa: float = 2.0
) -> np.ndarray:
    """Union of per-cycle window masks, applied cycle-locally."""
    mask = np.zeros(t.shape, dtype=bool)
    for c in cycles:
        _, w_lo, w_hi = detection_window(c, kappa=kappa)
        wrapped_time = cycle_local_time(t, c)
        mask |= (wrapped_time >= w_lo) & (wrapped_time <= w_hi)
    return mask


def window_mask(t: np.ndarray, w_lo: float, w_hi: float) -> np.ndarray:
    """Deprecated: global interval mask. Use combined_window_mask."""
    return (t >= w_lo) & (t <= w_hi)


def anomaly_amplitude(
    x: np.ndarray, r: np.ndarray, mask: np.ndarray
) -> float:
    """Definition 5: a = ||r * 1_W|| / ||x||, in [0, 1]."""
    if not mask.any():
        return 0.0
    num = float(np.linalg.norm(r[mask]))
    den = float(np.linalg.norm(x))
    if den == 0.0:
        return 0.0
    return float(np.clip(num / den, 0.0, 1.0))


def from_preset(
    name: str,
    *,
    amplitude: float = 1.0,
    phase0: float = 0.0,
    leading_ratio: float = 0.25,
    trailing_ratio: float = 0.25,
    separation_ratio: float = 1.0,
) -> GmGCycle:
    """Build a GmG cycle from a synodic preset plus user attributes.

    Parameters
    ----------
    name             : one of SYNODIC_PRESETS
    amplitude        : A
    phase0           : phi_0
    leading_ratio    : sigma_L / T
    trailing_ratio   : sigma_T / T
    separation_ratio : (mu_T - mu_L) / T
    """
    if name not in SYNODIC_PRESETS:
        raise KeyError(f"unknown preset {name!r}")
    T = SYNODIC_PRESETS[name]
    if not (0 < leading_ratio < 1 and 0 < trailing_ratio < 1):
        raise ValueError("sigma ratios must lie in (0, 1)")
    if separation_ratio <= 0:
        raise ValueError("separation_ratio must be positive")

    half = 0.5 * separation_ratio * T
    return GmGCycle(
        name=name,
        period=T,
        phase0=phase0,
        mu_L=-half,
        sigma_L=leading_ratio * T,
        mu_T=+half,
        sigma_T=trailing_ratio * T,
        amplitude=amplitude,
    )


def symmetrise(c: GmGCycle) -> GmGCycle:
    """Return the temporally symmetric version (sigma_L = sigma_T)."""
    sigma = 0.5 * (c.sigma_L + c.sigma_T)
    return replace(c, sigma_L=sigma, sigma_T=sigma)