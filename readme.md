# KAIROS

Cyclical null-space anomaly detection with Bayes-threshold window localisation
and quantum amplitude estimation.

KAIROS takes a time series and a declared set of cycles the series is expected
to follow. It computes the residual against the cyclical baseline, locates the
optimal detection window inside each cycle via the Bayes decision threshold,
and estimates the anomaly amplitude in that window using quantum amplitude
estimation on IQM hardware. Its output is a structured anomaly report with an
explicit hand-off payload for any downstream consumer.

KAIROS is standalone. It assumes nothing about where the data comes from or
where the report goes. It exposes a REST API, a WebSocket stream, a
command-line interface, and a plug-in adapter layer.

---

## 1. Why this module exists

Anomaly detectors fall into two families. Point-wise detectors answer *is this
point unusual relative to its neighbours*. Model-based detectors answer *is
this point unusual relative to the model we already believe*. The first family
is convenient and ignores structure; the second is precise and requires a
model.

KAIROS is model-based. The model is the set of cycles the user declares. A
detector that ignores cyclical structure flags the peak of every legitimate
cycle as an anomaly and misses the deviation that carries information. A
detector that knows the cycles subtracts them and examines what remains — the
null-space residual — which is where genuine anomalies live.

## Practical use cases

KAIROS applies wherever a time series carries cyclical structure the operator already trusts, and the question is not whether a point is unusual but whether a deviation from the expected cycle is real, localised, and worth acting on. In manufacturing and process telemetry, batch cycles, shift patterns, and scheduled maintenance provide the cyclical prior, and the anomaly is a deviation in yield, temperature, or throughput that occurs inside a specific window of the cycle rather than at the peak. In procurement and inventory, seasonal demand and reorder intervals define the baseline, and the anomaly is a sudden deviation in ordering behaviour, supplier lead time, or stock depletion, flagged before it becomes a stockout. In clinical operations, admission patterns follow weekly and seasonal rhythms, and the anomaly is a cluster that departs from the expected cadence inside a defined window — distinguishable from the routine Monday surge only because the cycle is known. In financial market microstructure, intraday and weekly cycles are well characterised, and the anomaly is a liquidity or volatility deviation that falls within a computable interval rather than at the open or close. In web and application telemetry, traffic follows diurnal and weekly patterns, and the anomaly is a latency or error excursion inside the window that would otherwise be masked by the natural variability of peak hours. In industrial quality control, production schedules supply the cycle, and the anomaly is a defect cluster whose temporal signature would be lost against the peak of every legitimate run. In each case KAIROS returns not merely a flag but the window in which the anomaly was localised, the residual amplitude relative to the declared baseline, and a structured payload that any downstream system — a decision engine, a causality analyser, an alerting pipeline, or an orchestrator reconciling multiple signals — can consume without further transformation.

FAIR WARNING: WEAKNESSES:

DATA BLINDNESS: The codebase does no input auditing or sanitization. Passing data strings containing NaNs, missing intervals, or structural gaps will cascade through the root-finding step and break the calculation layer

PHASE DEPENDENCY: If an enterprise deployment experiences a slight clock or shift drift, the code will continue to execute beautifully and output well-formed anomaly reports—but it will be calculating them against a fundamentally corrupted residual background.

THESE WEAKNESSES ARE WHY THE SOFTWARE IS INTENDED AS A RESEARCH PROJECT, AND NOT IN PRODUCTION!

---

## 2. Mathematical specification

### 2.1 The Gaussian-modulated Gaussian cycle

**Definition 1.** A GmG cycle is a tuple

\[
C = (T,\ \varphi_0,\ \mu_L,\ \sigma_L,\ \mu_T,\ \sigma_T,\ A),
\]

with envelope

\[
g_C(t) = A \exp\!\left(-\frac{(t-\mu_L)^2}{2\sigma_L^2}\right) \exp\!\left(-\frac{(t-\mu_T)^2}{2\sigma_T^2}\right).
\]

Here \(T\) is the period, \(\varphi_0\) the phase offset, \(\mu_L\) and
\(\sigma_L\) the centre and width of the leading Gaussian, \(\mu_T\) and
\(\sigma_T\) the centre and width of the trailing Gaussian, and \(A\) the
amplitude. All coordinates are in the same time units as \(t\).

**Lemma 1 (product of Gaussians).** For \(\sigma_L, \sigma_T > 0\),

\[
g_C(t) = A' \exp\!\left(-\frac{(t-\mu)^2}{2\sigma^2}\right),
\]

with

\[
\sigma^{-2} = \sigma_L^{-2} + \sigma_T^{-2}, \qquad
\mu = \sigma^2\!\left(\frac{\mu_L}{\sigma_L^2} + \frac{\mu_T}{\sigma_T^2}\right), \qquad
A' = A \exp\!\left(-\frac{(\mu_L-\mu_T)^2}{2(\sigma_L^2+\sigma_T^2)}\right).
\]

*Proof.* Complete the square in the exponent:

\[
\frac{(t-\mu_L)^2}{2\sigma_L^2} + \frac{(t-\mu_T)^2}{2\sigma_T^2}
= \frac{1}{2}\!\left(\frac{1}{\sigma_L^2} + \frac{1}{\sigma_T^2}\right)t^2
- \left(\frac{\mu_L}{\sigma_L^2} + \frac{\mu_T}{\sigma_T^2}\right)t
+ \left(\frac{\mu_L^2}{2\sigma_L^2} + \frac{\mu_T^2}{2\sigma_T^2}\right).
\]

Let \(1/\sigma^2 = 1/\sigma_L^2 + 1/\sigma_T^2\) and
\(\mu/\sigma^2 = \mu_L/\sigma_L^2 + \mu_T/\sigma_T^2\). The quadratic in \(t\)
becomes \(\frac{(t-\mu)^2}{2\sigma^2} + K\), where

\[
K = \frac{\mu_L^2}{2\sigma_L^2} + \frac{\mu_T^2}{2\sigma_T^2} - \frac{\mu^2}{2\sigma^2}.
\]

Expanding \(\mu^2/\sigma^2 = (\mu_L/\sigma_L^2 + \mu_T/\sigma_T^2)^2 \sigma^2\)
and simplifying using \(1/\sigma^2 = 1/\sigma_L^2 + 1/\sigma_T^2\):

\[
K = \frac{(\mu_L - \mu_T)^2}{2(\sigma_L^2 + \sigma_T^2)}.
\]

Therefore \(g_C(t) = A \exp(-K) \exp(-(t-\mu)^2/(2\sigma^2))\), giving
\(A' = A \exp(-K)\) as stated. \(\square\)

**Corollary 1.1.** The product-Gaussian mean lies between the two component
means: \(\min(\mu_L, \mu_T) \le \mu \le \max(\mu_L, \mu_T)\). The product width
is smaller than either component width: \(\sigma < \min(\sigma_L, \sigma_T)\).

*Proof.* The mean is a convex combination of \(\mu_L\) and \(\mu_T\) with
weights \(\sigma^2/\sigma_L^2\) and \(\sigma^2/\sigma_T^2\) that sum to one.
The width follows from \(\sigma^{-2} = \sigma_L^{-2} + \sigma_T^{-2} >
\sigma_L^{-2}\) and similarly for \(\sigma_T\). \(\square\)

### 2.2 The Bayes decision threshold

**Definition 2.** Under equal priors and equal costs, and assuming observations
drawn either from \(\mathcal{N}(\mu_L, \sigma_L^2)\) (signal) or
\(\mathcal{N}(\mu_T, \sigma_T^2)\) (noise), the optimal decision threshold
\(x^*\) minimises the expected misclassification rate
\[
R(x) = \tfrac{1}{2}\bigl(1 - P_{\mathrm{TP}}(x)\bigr) + \tfrac{1}{2} P_{\mathrm{FP}}(x),
\]
where \(P_{\mathrm{TP}}(x) = P(X \ge x \mid \text{signal})\)
and \(P_{\mathrm{FP}}(x) = P(X \ge x \mid \text{noise})\).

**Theorem 1 (Bayes threshold, existence and uniqueness).** The minimiser \(x^*\)
of \(R(x)\) is the unique root on \([\min(\mu_L,\mu_T), \max(\mu_L,\mu_T)]\) of
\[
\frac{1}{\sigma_L}\,\phi\!\left(\frac{x-\mu_L}{\sigma_L}\right)
=
\frac{1}{\sigma_T}\,\phi\!\left(\frac{x-\mu_T}{\sigma_T}\right),
\tag{$\ast$}
\]
where \(\phi\) is the standard normal PDF. When \(\sigma_L = \sigma_T\),
\(x^* = (\mu_L + \mu_T)/2\).

*Proof.* Differentiating \(R\):

\[
R'(x) = \tfrac{1}{2} \frac{d}{dx}\bigl(1 - P_{\mathrm{TP}}(x)\bigr) + \tfrac{1}{2}\frac{d}{dx}P_{\mathrm{FP}}(x).
\]

Using \(P_{\mathrm{TP}}(x) = 1 - \Phi((x-\mu_L)/\sigma_L)\) and
\(P_{\mathrm{FP}}(x) = 1 - \Phi((x-\mu_T)/\sigma_T)\), we have
\((1-P_{\mathrm{TP}})'(x) = f_L(x)\) and \(P_{\mathrm{FP}}'(x) = -f_T(x)\),
where \(f_L, f_T\) are the signal and noise densities. Therefore

\[
R'(x) = \tfrac{1}{2}\bigl(f_L(x) - f_T(x)\bigr).
\]

Setting \(R'(x^*) = 0\) yields \(f_L(x^*) = f_T(x^*)\), which is equation
\((\ast)\). Uniqueness: on the open interval
\((\min(\mu_L,\mu_T), \max(\mu_L,\mu_T))\), the log-likelihood ratio

\[
\ell(x) = \log \frac{f_L(x)}{f_T(x)}
       = \log\frac{\sigma_T}{\sigma_L}
       - \frac{(x-\mu_L)^2}{2\sigma_L^2}
       + \frac{(x-\mu_T)^2}{2\sigma_T^2}
\]

has derivative

\[
\ell'(x) = -\frac{x-\mu_L}{\sigma_L^2} + \frac{x-\mu_T}{\sigma_T^2}.
\]

Suppose without loss of generality that \(\mu_L < \mu_T\) and
\(x \in (\mu_L, \mu_T)\). Then \(x - \mu_L > 0\) and \(x - \mu_T < 0\), so
\(-(x-\mu_L)/\sigma_L^2 < 0\) and \((x-\mu_T)/\sigma_T^2 < 0\), giving
\(\ell'(x) < 0\). Therefore \(\ell\) is strictly decreasing on the interval
and the ratio \(f_L/f_T\) attains the value \(1\) at most once. Existence
follows from

\[
\lim_{x \to \mu_L^+} \frac{f_L(x)}{f_T(x)} > 1
\quad\text{and}\quad
\lim_{x \to \mu_T^-} \frac{f_L(x)}{f_T(x)} < 1
\]

when \(\sigma_L = \sigma_T\), and from the sign change of \(f_L - f_T\) on
the endpoints in the general case. When \(\sigma_L = \sigma_T\), symmetry
of \(f_L\) and \(f_T\) about the midpoint gives \(x^* = (\mu_L+\mu_T)/2\). \(\square\)

**Remark 1.** An earlier version of this document framed the threshold as a
Nash equilibrium of a two-player game between "signal" and "noise" players.
That framing is heuristic and, if taken literally, does not yield a
well-defined game: the signal player's payoff is strictly decreasing in
\(x\), the noise player's is strictly increasing, so best responses are
\(x = -\infty\) and \(x = +\infty\) respectively. No finite-strategy
equilibrium exists. The correct mathematical content is the Bayes decision
threshold derived in Theorem 1 above, and the implementation computes this
threshold. The function `bayes_threshold` returns it; the alias
`nash_detection_point` is retained for backward compatibility.

### 2.3 The detection window

**Definition 3.** Given the Bayes threshold \(x^*\), the product width
\(\sigma\) from Lemma 1, and a user-specified width parameter \(\kappa > 0\),
the detection window is

\[
W^*(\kappa) = [x^* - \kappa\sigma,\ x^* + \kappa\sigma].
\]

**Remark 2.** \(W^*(\kappa)\) is not implied by Theorem 1. Theorem 1
characterises the optimal point threshold; the window is a user choice
trading coverage (larger \(\kappa\), more of the signal but also more noise)
against specificity (smaller \(\kappa\), fewer false alarms). The default
\(\kappa = 2\) gives approximately two product-Gaussian standard deviations
on each side of the threshold.

### 2.4 The residual and the anomaly amplitude

**Definition 4.** For a series \(x(t)\) sampled on a uniform grid
\(t_1, \ldots, t_n\) and a stack of cycles
\(\mathcal{C} = \{C_1, \ldots, C_m\}\), the baseline is

\[
b(t) = \sum_{j=1}^{m} g_{C_j}\!\big(\psi_j(t)\big),
\qquad
\psi_j(t) = \left(\frac{t - \varphi_{0,j}}{T_j} + \tfrac{1}{2}\right) \bmod 1 - \tfrac{1}{2}\ \cdot\ T_j,
\]

where \(\psi_j(t) \in [-T_j/2, T_j/2)\) is the **cycle-local time**. The
residual is \(r(t) = x(t) - b(t)\).

**Definition 5.** The anomaly amplitude in window mask \(M \subseteq \{1,
\ldots, n\}\) is

\[
a = \frac{\|r \cdot \mathbb{1}_M\|_2}{\|x\|_2} \in [0, 1].
\]

**Lemma 3 (null expectation).** Under the null hypothesis
\(x(t) = b(t) + \varepsilon(t)\) with \(\varepsilon \sim \mathcal{N}(0,
\sigma_\varepsilon^2)\), \(\mathbb{E}[r(t)] = 0\), and \(a\) converges in
probability to \(0\) as \(n \to \infty\) with rate \(O(n^{-1/2})\) provided
the noise is independent across samples and the mask \(M\) has measure
proportional to \(n\).

*Proof.* \(r(t) = \varepsilon(t)\), so \(\|r \cdot \mathbb{1}_M\|_2^2 =
\sum_{i \in M} \varepsilon_i^2\). By the law of large numbers,
\(\frac{1}{|M|}\sum_{i\in M}\varepsilon_i^2 \to \sigma_\varepsilon^2\)
almost surely. Meanwhile \(\|x\|_2^2 = \sum_i b_i^2 + o(n)\) for the signal
component. Therefore \(a \to \sqrt{\sigma_\varepsilon^2 \cdot |M| / \sum_i b_i^2}\).
Under an anomaly \(x(t) = b(t) + \varepsilon(t) + \delta(t)\) with
\(\delta \cdot \mathbb{1}_M\) having norm exceeding
\(\sqrt{\sigma_\varepsilon^2 |M|}\), the ratio \(a\) converges to a strictly
positive value. \(\square\)

### 2.5 Quantum amplitude estimation

**Definition 6.** The windowed residual is encoded as a quantum state

\[
|r_W\rangle = \frac{1}{\|r_W\|}\sum_{i \in M} r_W(i) \, |i\rangle,
\]

where \(r_W(i) = r(t_i)\) for \(i \in M\). An oracle marks indices \(i\) for
which \(|r_W(i)| > \theta\); the marked subspace has probability
\(p = \|r_W^{\text{marked}}\|^2 / \|r_W\|^2\).

**Definition 7.** The quantum anomaly amplitude is
\(a_{\mathrm{QAE}} = (\|r_W\|/\|x\|)\sqrt{\hat{p}}\), where \(\hat{p}\) is the
QAE estimate of \(p\).

**Theorem 2 (Brassard–Høyer–Mosca–Tapp, 2002).** With \(m\) evaluation
qubits, the QAE estimate \(\hat{p}\) satisfies

\[
|\hat{p} - p| \le \frac{2\pi\sqrt{p(1-p)}}{2^m} + \frac{\pi^2}{2^{2m}}
\]

with probability at least \(8/\pi^2\).

The implementation uses Qiskit with `iqm.qiskit_iqm.IQMProvider` for IQM
submission and `qiskit_aer` for simulation. The classical estimate is
computed in all cases as a comparison baseline; the quantum path is an
accelerator and a verification, not a dependency.

### 2.6 Implementation notes

- The cycle-local time \(\psi_j(t)\) is used for both the baseline
  computation and the window mask. This ensures that multi-cycle series
  correctly identify anomalies in any cycle, not only the first.
- The default marking threshold is
  \(\theta = \overline{|r_W|} + 2\,\mathrm{std}(|r_W|)\), a statistical
  outlier criterion rather than a fixed percentile. The user may override
  this with a fixed threshold.

---

## 3. Installation

### 3.1 Requirements

- Python ≥ 3.10
- No native compilation

```bash
git clone <this-repo> kairos && cd kairos
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Quantum packages (`qiskit`, `qiskit-aer`, `qiskit-algorithms`, `iqm-client`)
are optional. Without them, KAIROS runs the classical amplitude estimate and
reports `backend = "classical"`.

### 3.2 IQM access

Register at `resonance.iqm.tech` and generate an API token from the account
profile. The Starter tier provides up to 30 credits per month on selected IQM
quantum computers. Export the token:

```bash
export IQM_TOKEN="<your-token>"
```

### 3.3 Verification

```bash
python -c "from core import from_preset, bayes_threshold; \
           c = from_preset('mercury_sun'); \
           print(round(bayes_threshold(c), 6))"
# 0.0
```

Full smoke test:

```bash
python test_smoke.py
```

All assertions should pass. See Section 6 for expected output.

---

## 4. Running

### 4.1 Server and UI

```bash
python run.py                    # http://127.0.0.1:8300
```

The browser UI seeds a synthetic quarterly cycle on load. Click **Run
detection**. OpenAPI documentation: `http://127.0.0.1:8300/docs`.

### 4.2 Command line

```bash
python cli.py series.csv --preset mercury_sun --threshold 0.05 --quantum
```

`series.csv` is a two-column file: `t,value` per line, no header.

---

## 5. API

### 5.1 `POST /api/detect`

Request:

```json
{
  "series": [0.1, 0.2, ...],
  "t": [0.0, 0.1, ...],
  "cycles": [{
    "name": "quarterly", "period": 90.0, "phase0": 0,
    "mu_L": -22.5, "sigma_L": 22.5,
    "mu_T":  22.5, "sigma_T": 22.5,
    "amplitude": 0.4
  }],
  "kappa": 2.0,
  "threshold": 0.05,
  "qae_eval_qubits": 6,
  "quantum": true
}
```

Response:

```json
{
  "kind": "kairos.anomaly_report.v1",
  "cycles": [{
    "name": "quarterly", "period": 90.0,
    "product_mean": 0.0, "product_sigma": 15.91,
    "product_amplitude": 0.4,
    "threshold": 0.0,
    "window_lo": -31.82, "window_hi": 31.82
  }],
  "window": {"x_star": 0.0, "lo": -31.82, "hi": 31.82},
  "amplitude": {
    "classical": 0.312, "quantum": 0.311,
    "backend": "qiskit-aer", "quantum_error": 0.002
  },
  "residual": {"norm": 1.83, "series_norm": 5.85},
  "flagged": true,
  "threshold": 0.05
}
```

### 5.2 `GET /api/cycles/presets`

Returns the synodic period table.

### 5.3 `POST /api/cycles/evaluate`

Returns the baseline for a given time grid and cycle stack.

### 5.4 `POST /api/adapters/register`

Register any external software:

```json
{"name": "downstream", "base_url": "http://localhost:8000", "path": "/api/ingest"}
```

### 5.5 `POST /api/adapters/{name}/call`

Dispatch the hand-off payload.

### 5.6 `WS /ws/detect`

Live detection stream. Send `{"type": "detect", ...}` with the same fields as
the REST request.

---

## 6. Verification trace

Running `python test_smoke.py` on a clean installation produces:

```
[PASS] Lemma 1: product Gaussian matches envelope (max abs diff < 1e-12)
[PASS] Theorem 1: Bayes threshold at midpoint for symmetric cycle
[PASS] Theorem 1: likelihoods equal at threshold (rel err < 1e-10)
[PASS] Baseline periodic: b(t) = b(t + kT) for k=1,2
[PASS] Window mask: cycle-local wrapping across multiple cycles
[PASS] Detector: anomaly flagged (amplitude ~ 0.105)
[PASS] Detector: anomaly not flagged (amplitude ~ 0.021)
All assertions passed.
```

The numerical values for the last two lines depend on the amplitude of the
synthetic cycle, the noise level, and the spike magnitude used inside
`test_smoke._make_series`. The pass criterion is that the flagged case
exceeds the `0.05` threshold and the unflagged case is below it; the exact
values are not asserted, only the flag.

---

## 7. Integration patterns

The adapter layer is a generic HTTP contract: any software that accepts a POST
of the `kairos.anomaly_report.v1` payload and returns JSON is a valid
downstream. Examples:

**Pattern A — KAIROS upstream of a decision engine.** The anomaly report
supplies the window and residual statistics; the decision engine uses them as
the time-bounded input to its own phases.

**Pattern B — KAIROS downstream of a causality analyser.** The causality tool
produces a candidate critical-juncture window; KAIROS validates it against the
declared cyclical prior and flags spurious junctures.

**Pattern C — Parallel to a swarm coordinator.** Both KAIROS and any other
scoring module report to the coordinator, which reconciles the cyclical
anomaly score with other signals.

**Pattern D — Streaming to a coordination bus.** WebSocket updates feed the
downstream coordinator continuously; each tick is a fresh report.

Example registration:

```python
from adapters import registry
registry.register("decision_engine", "http://localhost:8000", "/api/ingest")
report = detect(series, t, cycles)
registry.call("decision_engine", report.to_handoff())
```

---

## 8. Limitations

1. **The cycle prior is user-supplied.** KAIROS does not learn cycles. If the
   declared stack is wrong, the residual is wrong.
2. **The detection threshold is the Bayes threshold under equal priors and
   equal costs.** Unequal priors or costs require a modification not present
   in this implementation.
3. **The window is a design choice.** Theorem 1 characterises the optimal
   point threshold, not the optimal interval; the width parameter \(\kappa\)
   trades coverage against specificity.
4. **QAE error is bounded, not eliminated.** The Brassard bound holds with
   probability \(8/\pi^2\); hardware noise adds error not captured by the
   ideal bound.
5. **The quantum path is an accelerator.** On current hardware, classical
   amplitude estimation is competitive at the problem sizes KAIROS handles.
6. **The synodic presets are illustrative.** They are physically real
   periods, not a claim about causation.
7. **Data validity is not audited.** KAIROS does not check whether the input
   series is well-formed, free of NaNs, or sampled at the claimed cadence.

---

## 9. License

MIT.
