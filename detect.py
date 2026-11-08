"""
detect.py — the KAIROS anomaly detection module.

Standalone. Consumes a time series and a declared cycle stack, produces an
anomaly report with an explicit hand-off payload for any downstream consumer.

Public surface:
    detect(series, t, cycles, ...)  -> AnomalyReport
    AnomalyReport.to_handoff()      -> dict
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from core import (
    GmGCycle,
    bayes_threshold,
    detection_window,
    product_parameters,
    combined_window_mask,
    residual,
    anomaly_amplitude,
)
from quantum_backend import estimate_amplitude


@dataclass
class AnomalyReport:
    cycles: list[dict]
    x_star: float
    window: tuple[float, float]
    classical_amplitude: float
    quantum_amplitude: float
    quantum_backend: str
    quantum_error: float | None
    residual_norm: float
    series_norm: float
    flagged: bool
    threshold: float
    extra: dict = field(default_factory=dict)

    def to_handoff(self) -> dict:
        return {
            "kind": "kairos.anomaly_report.v1",
            "cycles": self.cycles,
            "window": {
                "x_star": self.x_star,
                "lo": self.window[0],
                "hi": self.window[1],
            },
            "amplitude": {
                "classical": self.classical_amplitude,
                "quantum": self.quantum_amplitude,
                "backend": self.quantum_backend,
                "quantum_error": self.quantum_error,
            },
            "residual": {
                "norm": self.residual_norm,
                "series_norm": self.series_norm,
            },
            "flagged": self.flagged,
            "threshold": self.threshold,
            "extra": self.extra,
        }


def detect(
    series: Sequence[float],
    t: Sequence[float],
    cycles: Sequence[GmGCycle],
    *,
    kappa: float = 2.0,
    threshold: float = 0.05,
    qae_eval_qubits: int = 6,
    qae_shots: int = 2048,
    quantum: bool = True,
    iqm_token: str | None = None,
    iqm_server: str = "https://resonance.iqm.tech",
    iqm_backend: str | None = None,
    mark_threshold: float | None = None,
) -> AnomalyReport:
    """Run the full detection pipeline on one series.

    Parameters
    ----------
    series             : observed values x(t), 1-D
    t                  : time grid, 1-D, same length as series
    cycles             : declared cycle stack
    kappa              : window half-width in product-sigma units
    threshold          : flag threshold on the classical amplitude
    qae_eval_qubits    : evaluation qubits for the QAE circuit
    qae_shots          : reserved for future shot-based backends
    quantum            : run the QAE path (True) or classical only (False)
    iqm_token          : IQM API token; if None, Aer or classical fallback
    iqm_server         : IQM Resonance endpoint
    iqm_backend        : specific IQM backend name, or None for default
    mark_threshold     : absolute marked-amplitude threshold; if None,
                         use mean(|r_W|) + 2 std(|r_W|)
    """
    x = np.asarray(series, dtype=float)
    tt = np.asarray(t, dtype=float)
    if x.ndim != 1 or tt.ndim != 1 or x.shape != tt.shape:
        raise ValueError("series and t must be 1-D arrays of equal length")
    if x.size == 0:
        raise ValueError("series and t must be non-empty")
    if not cycles:
        raise ValueError("at least one cycle is required")
    if kappa <= 0:
        raise ValueError("kappa must be positive")
    if not (0.0 < threshold < 1.0):
        raise ValueError("threshold must lie in (0, 1)")
    if not (2 <= qae_eval_qubits <= 20):
        raise ValueError("qae_eval_qubits must lie in [2, 20]")

    cycle_diag: list[dict] = []
    for c in cycles:
        mu, sigma, A = product_parameters(c)
        x_star, w_lo, w_hi = detection_window(c, kappa=kappa)
        cycle_diag.append({
            "name": c.name,
            "period": c.period,
            "product_mean": mu,
            "product_sigma": sigma,
            "product_amplitude": A,
            "threshold": x_star,
            "window_lo": w_lo,
            "window_hi": w_hi,
        })

    mask = combined_window_mask(tt, cycles, kappa=kappa)
    r = residual(x, tt, cycles)
    a_classical = anomaly_amplitude(x, r, mask)

    if quantum and mask.any():
        q = estimate_amplitude(
            x=x, r=r, mask=mask,
            n_eval_qubits=qae_eval_qubits,
            shots=qae_shots,
            mark_threshold=mark_threshold,
            iqm_token=iqm_token,
            iqm_server=iqm_server,
            iqm_backend=iqm_backend,
        )
        a_quantum = q["amplitude"]
        backend_name = q["backend"]
        qerr = q.get("error")
    else:
        a_quantum, backend_name, qerr = a_classical, "classical-skip", None

    w_lo = min(d["window_lo"] for d in cycle_diag)
    w_hi = max(d["window_hi"] for d in cycle_diag)
    x_star_mean = float(np.mean([d["threshold"] for d in cycle_diag]))

    return AnomalyReport(
        cycles=cycle_diag,
        x_star=x_star_mean,
        window=(float(w_lo), float(w_hi)),
        classical_amplitude=float(a_classical),
        quantum_amplitude=float(a_quantum),
        quantum_backend=backend_name,
        quantum_error=qerr,
        residual_norm=float(np.linalg.norm(r[mask])) if mask.any() else 0.0,
        series_norm=float(np.linalg.norm(x)),
        flagged=bool(a_classical >= threshold),
        threshold=float(threshold),
    )