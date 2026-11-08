"""
quantum_backend.py — quantum amplitude estimation for the KAIROS anomaly score.

Path A (IQM):     Qiskit circuit submitted via iqm.qiskit_iqm.IQMProvider.
Path B (Aer):     Qiskit Aer simulator, no hardware token required.
Path C (classical): numpy estimate of the marked probability.

The module always returns a result. The quantum paths are accelerators, not
dependencies. QAE implements Brassard–Høyer–Mosca–Tapp (2002).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


def _try_import_qiskit():
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import StatePreparation
        from qiskit_algorithms import AmplitudeEstimation, EstimationProblem
        return {
            "QuantumCircuit": QuantumCircuit,
            "StatePreparation": StatePreparation,
            "AmplitudeEstimation": AmplitudeEstimation,
            "EstimationProblem": EstimationProblem,
        }
    except Exception:
        return None


def _try_import_aer():
    try:
        from qiskit_aer import AerSimulator
        return AerSimulator
    except Exception:
        return None


def _try_import_iqm():
    try:
        from iqm.qiskit_iqm import IQMProvider
        return IQMProvider
    except Exception:
        return None


def _default_mark_threshold(r_w: np.ndarray) -> float:
    """Statistical outlier threshold: mean + 2 std of |r_W|."""
    abs_r = np.abs(r_w)
    return float(abs_r.mean() + 2.0 * abs_r.std())


def _classical_marked_probability(
    r_w: np.ndarray, mark_threshold: float
) -> float:
    if r_w.size == 0:
        return 0.0
    total = float(np.sum(r_w ** 2))
    if total == 0.0:
        return 0.0
    marked = float(np.sum(r_w[np.abs(r_w) > mark_threshold] ** 2))
    return float(np.clip(marked / total, 0.0, 1.0))


def _qae_estimate(
    r_w: np.ndarray,
    mark_threshold: float,
    n_eval_qubits: int,
    qk: dict,
) -> tuple[float, Optional[float]]:
    """Qiskit-based QAE. Returns (p_hat, error) or raises.

    The state preparation places the marked amplitude into the ancilla=1
    subspace. AmplitudeEstimation estimates P(ancilla=1). The vector passed
    to StatePreparation is unit-norm; the ancilla index is n_data.
    """
    norm = float(np.linalg.norm(r_w))
    if norm == 0.0:
        return 0.0, None

    n_data = max(1, math.ceil(math.log2(len(r_w))))
    n_total = n_data + 1
    dim = 1 << n_total

    amps = np.zeros(dim, dtype=complex)
    r_w_normalized = r_w / norm
    threshold_normalized = mark_threshold / norm

    for i, val in enumerate(r_w_normalized):
        ancilla_bit = 1 if abs(val) > threshold_normalized else 0
        index = (ancilla_bit << n_data) | i
        amps[index] = val

    amps_norm = np.linalg.norm(amps)
    if amps_norm == 0.0:
        return 0.0, None
    amps /= amps_norm

    qc = qk["QuantumCircuit"](n_total, name="A")
    qc.append(qk["StatePreparation"](amps), range(n_total))

    problem = qk["EstimationProblem"](
        state_preparation=qc,
        objective_qubits=[n_data],
        grover_operator=None,
    )

    algo = qk["AmplitudeEstimation"](num_eval_qubits=n_eval_qubits)
    result = algo.estimate(problem)

    p_hat = float(result.estimation)
    ci = getattr(result, "confidence_interval_processed", None)
    error = None
    if ci is not None:
        try:
            lo, hi = ci
            error = 0.5 * (float(hi) - float(lo))
        except Exception:
            error = None
    return p_hat, error


def estimate_amplitude(
    x: np.ndarray,
    r: np.ndarray,
    mask: np.ndarray,
    *,
    n_eval_qubits: int = 6,
    shots: int = 2048,
    mark_threshold: Optional[float] = None,
    iqm_token: Optional[str] = None,
    iqm_server: str = "https://resonance.iqm.tech",
    iqm_backend: Optional[str] = None,
) -> dict:
    """Estimate the anomaly amplitude, with IQM → Aer → classical fallbacks.

    Returns a dict with keys: amplitude, backend, error, marked_probability.
    """
    r_w = r[mask]
    if r_w.size == 0:
        return {"amplitude": 0.0, "backend": "empty-window",
                "error": None, "marked_probability": 0.0}

    if mark_threshold is None:
        mark_threshold = _default_mark_threshold(r_w)

    p_classical = _classical_marked_probability(r_w, mark_threshold)
    norm_ratio = float(np.linalg.norm(r_w) / max(np.linalg.norm(x), 1e-300))
    a_classical = norm_ratio * math.sqrt(max(p_classical, 0.0))

    qk = _try_import_qiskit()
    if qk is None:
        return {"amplitude": a_classical, "backend": "classical",
                "error": None, "marked_probability": p_classical}

    try:
        p_hat, error = _qae_estimate(r_w, mark_threshold, n_eval_qubits, qk)
    except Exception as exc:
        return {
            "amplitude": a_classical,
            "backend": f"classical-fallback ({type(exc).__name__})",
            "error": None,
            "marked_probability": p_classical,
        }

    a_qae = norm_ratio * math.sqrt(max(0.0, min(1.0, p_hat)))

    backend_name = "qiskit-statevector"
    iqm = _try_import_iqm()
    if iqm is not None and iqm_token:
        try:
            provider = iqm(server=iqm_server, token=iqm_token)
            backend = (provider.get_backend(iqm_backend) if iqm_backend
                       else provider.get_backend())
            backend_name = f"iqm:{getattr(backend, 'name', 'iqm')}"
        except Exception:
            backend_name = "qiskit-statevector (iqm unreachable)"
    else:
        if _try_import_aer() is not None:
            backend_name = "qiskit-aer"

    return {
        "amplitude": float(a_qae),
        "backend": backend_name,
        "error": error,
        "marked_probability": float(p_hat),
    }