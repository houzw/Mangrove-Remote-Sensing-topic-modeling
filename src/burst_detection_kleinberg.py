"""Kleinberg (2003) two-state burst detection — Binomial (volume-sensitive) form.

A faithful, self-contained re-implementation of the burst-detection automaton
for *bibliometric share series*, adapted from Kleinberg's Poisson gap model to a
Binomial observation model so the hidden state is defined on the topic's
**share** rather than its raw count (which would otherwise be swamped by the
corpus's own exponential growth).

Model
-----
For one topic, per year ``y`` we observe ``N[y]`` documents assigned to the
topic out of ``T[y]`` documents published that year.  The observation model is

    N[y] ~ Binomial(T[y], p_{s[y]}),   s[y] in {0, 1},   p0 < p1,

where ``s[y]=0`` is the baseline state (share ``p0``) and ``s[y]=1`` the burst
state (share ``p1``).  The state cost is the negative log-likelihood

    c_s(y) = -[ N[y] ln p_s + (T[y]-N[y]) ln(1-p_s) ],

with the ``0 * log 0 = 0`` convention.  A state transition costs ``gamma``.

Estimation and decoding are two *separate* stages, and the write-up in the
manuscript follows this split exactly.

1. **Rate estimation (``estimate_rates``).**  ``p0`` and ``p1`` are estimated per
   topic by a *regularized hard-EM* over the years.  The E-step is a per-year
   maximum-likelihood assignment of that year to the cheaper of the two states
   *given the current rates* -- ``mask1 = c1 < c0`` -- so it is a pointwise
   assignment and deliberately does **not** consult the transition cost
   ``gamma``; the M-step re-estimates both rates as Jeffreys-smoothed weighted
   means.  This is therefore *not* Viterbi training / segmental k-means; the
   transition structure is not part of the likelihood being maximized here.
   Jeffreys (Beta(1/2,1/2)) pseudocounts keep both rates strictly in (0,1) so
   ``W`` is always finite, a separation constraint ``p1 >= 2 * p0`` excludes the
   degenerate all-baseline fixed point (where ``p1`` would be undefined), and
   ``N_RESTARTS`` starts in total -- one deterministic (the highest-share
   year, seeded as burst) and the remaining ``N_RESTARTS - 1`` seeded
   random -- with a state-path-stability convergence test, guard against
   local optima.

2. **Path decoding (``viterbi_path``).**  The reported state path -- and hence
   the burst intervals -- is the minimum-cost path of the two-state automaton
   found by the Viterbi algorithm under the *estimated* ``(p0, p1)`` and the
   transition cost ``gamma`` (2 states, so O(Y) and trivially cheap).  ``gamma``
   therefore controls only the segmentation, not the per-topic rate estimates.

Empty calendar years never enter either stage: callers pass only the
corpus-covered years (``T[y] > 0``), so there is no zero-denominator bin that
the automaton could traverse at zero cost.  Consequently a "contiguous run" in
``burst_intervals`` is contiguous in the *supplied year sequence*; when that
sequence has internal gaps a run's calendar span can exceed its number of
observed bins, so interval lengths must be read against the supplied years.

The burst *strength* of an interval is the log-likelihood-ratio of the burst
state over the baseline state summed across the interval,

    W = sum_{y in interval} [ N[y] ln(p1/p0) + (T[y]-N[y]) ln((1-p1)/(1-p0)) ],

i.e. Kleinberg's notion of "burst weight": how much better the burst state
explains that stretch of data than the baseline does.

Usage (self-test)::

    pixi run python -m src.burst_detection_kleinberg
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

# --- numerical guard for log terms ------------------------------------------
# Backstop only: with Jeffreys smoothing (see _weighted_share) every rate is
# already strictly in (0,1), so this clip no longer sets W for cold-start
# topics.  Retained as a defensive guard against log(0)/log(1) at machine edge.
_LOG_FLOOR = 1e-12

# --- hard-EM regularization constants ----------------------------------------
# Jeffreys prior Beta(1/2, 1/2): posterior mean (x + 0.5) / (n + 1.0).  This
# keeps p0 and p1 strictly inside (0,1) even for a cold-start topic whose
# baseline years have zero documents, so ln(p1/p0) — and hence W — is always
# finite and data-driven rather than set by an arbitrary _LOG_FLOOR.
JEFFREYS_PSEUDOCOUNT = 0.5
# Separation constraint p1 >= SEP * p0, applied in every M-step.  This excludes
# the degenerate all-baseline fixed point (where p1 would be undefined) and
# keeps the log-likelihood-ratio away from 0/0.
SEP = 2.0
# Total number of hard-EM starts, not the number of random restarts: the
# deterministic highest-share-year start is placed first and the loop then
# appends random ones until this count is reached, so the split is
# 1 deterministic + (N_RESTARTS - 1) random.  The seed is fixed, so the
# multi-start is reproducible; the run with the lowest negative
# log-likelihood is kept.
N_RESTARTS = 10
EM_SEED = 0


def _neg_binom_ll(N: np.ndarray, T: np.ndarray, p: float) -> np.ndarray:
    """Negative log-likelihood of N ~ Binomial(T, p), up to a constant term.

    Returns an array aligned with ``N``/``T``; lower = better fit.
    """
    p = float(np.clip(p, _LOG_FLOOR, 1.0 - _LOG_FLOOR))
    q = 1.0 - p
    # N * log(p), with 0*log0=0
    a = np.where(N > 0, N.astype(float) * np.log(p), 0.0)
    # (T - N) * log(1-p), with 0*log0=0
    b = np.where((T - N) > 0, (T - N).astype(float) * np.log(q), 0.0)
    return -(a + b)


def _weighted_share(N: np.ndarray, T: np.ndarray, mask: np.ndarray) -> float:
    """Jeffreys-smoothed weighted-mean share over the masked years.

    Posterior mean of a Beta(1/2, 1/2) prior on Binomial(sum N, sum T):

        p = (sum N[mask] + 0.5) / (sum T[mask] + 1.0).

    The pseudocounts keep the estimate strictly in (0,1), so a cold-start topic
    whose baseline years are all zero yields a small-but-positive ``p0`` instead
    of an exact 0 (which would make ``ln(p1/p0)`` blow up).
    """
    num = float(N[mask].sum()) + JEFFREYS_PSEUDOCOUNT
    den = float(T[mask].sum()) + 2.0 * JEFFREYS_PSEUDOCOUNT
    if den <= 0:
        return JEFFREYS_PSEUDOCOUNT / (2.0 * JEFFREYS_PSEUDOCOUNT)
    return float(num / den)


def _em_cost(N: np.ndarray, T: np.ndarray, p0: float, p1: float, mask1: np.ndarray) -> float:
    """Negative log-likelihood of a hard assignment (lower = better)."""
    c0 = _neg_binom_ll(N, T, p0)
    c1 = _neg_binom_ll(N, T, p1)
    return float(np.where(mask1, c1, c0).sum())


def _hard_em_once(
    N: np.ndarray,
    T: np.ndarray,
    init_mask1: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> Tuple[float, float, np.ndarray, float]:
    """One hard-EM run from ``init_mask1``.

    E-step assigns each year to the cheaper state *pointwise* under the current
    rates (``0 * log 0`` convention); it does not use the transition cost, which
    enters only at decoding time (``viterbi_path``).  M-step re-estimates the two
    Jeffreys-smoothed shares under the separation constraint ``p1 >= SEP * p0``.
    Converges when the state path is stable (``new_mask1 == mask1``) *and* the
    parameters are within ``tol``.

    Returns ``(p0, p1, mask1, cost)``.
    """
    N = np.asarray(N, dtype=float)
    T = np.asarray(T, dtype=float)
    share = np.divide(N, T, out=np.zeros_like(N), where=T > 0)
    top = int(np.argmax(share))
    low = int(np.argmin(share))

    mask1 = init_mask1.astype(bool).copy()
    p0 = _weighted_share(N, T, ~mask1)
    p1 = max(_weighted_share(N, T, mask1), SEP * p0)

    for _ in range(max_iter):
        c0 = _neg_binom_ll(N, T, p0)
        c1 = _neg_binom_ll(N, T, p1)
        new_mask1 = c1 < c0
        # keep both states non-empty so p0 and p1 stay well-defined
        if not new_mask1.any():
            new_mask1[top] = True
        if new_mask1.all():
            new_mask1[low] = False
        new_p0 = _weighted_share(N, T, ~new_mask1)
        new_p1 = max(_weighted_share(N, T, new_mask1), SEP * new_p0)

        path_stable = bool(np.array_equal(new_mask1, mask1))
        params_stable = abs(new_p0 - p0) < tol and abs(new_p1 - p1) < tol
        p0, p1, mask1 = new_p0, new_p1, new_mask1
        if path_stable and params_stable:
            break

    return float(p0), float(p1), mask1, _em_cost(N, T, p0, p1, mask1)


def estimate_rates(
    N: np.ndarray,
    T: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-9,
    n_restarts: int = N_RESTARTS,
    seed: int = EM_SEED,
) -> Tuple[float, float]:
    """Estimate baseline share ``p0`` and burst share ``p1`` by regularized hard-EM.

    Hard-EM alternates a per-year ML assignment (E-step) with Jeffreys-smoothed
    weighted-mean re-estimation under the separation constraint ``p1 >= SEP * p0``
    (M-step).  Because the all-baseline path is excluded by the constraint and
    every rate is strictly in (0,1), the estimator cannot collapse to a
    degenerate point.  ``n_restarts`` starts in total -- the deterministic
    highest-share-year start plus seeded random restarts -- are run and the
    lowest-cost solution is kept, so a bad local optimum is not silently
    reported.

    Returns ``(p0, p1)`` with ``p1 >= SEP * p0``; if the topic has no documents
    at all the pair degenerates to ``(0.0, 0.0)``.
    """
    N = np.asarray(N, dtype=float)
    T = np.asarray(T, dtype=float)
    if T.sum() <= 0:
        return 0.0, 0.0
    n = len(N)
    if n == 0:
        return 0.0, 0.0

    share = np.divide(N, T, out=np.zeros_like(N), where=T > 0)
    top = int(np.argmax(share))

    # deterministic start (highest-share year = burst) + seeded random restarts
    inits = []
    m = np.zeros(n, dtype=bool)
    m[top] = True
    inits.append(m)
    rng = np.random.default_rng(seed)
    while len(inits) < n_restarts:
        m = rng.random(n) < 0.2
        if not m.any():
            m[top] = True
        if m.all():
            m[top] = False
        inits.append(m)

    best = None
    for m0 in inits:
        p0, p1, mask1, cost = _hard_em_once(N, T, m0, max_iter, tol)
        if best is None or cost < best[0]:
            best = (cost, p0, p1, mask1)

    _, p0, p1, _ = best
    return float(p0), float(p1)


def viterbi_path(
    N: np.ndarray,
    T: np.ndarray,
    p0: float,
    p1: float,
    gamma: float = 1.0,
) -> np.ndarray:
    """Minimum-cost state path under the two-state cost model.

    Returns an int array of the same length as ``N`` with values in ``{0, 1}``.
    """
    N = np.asarray(N, dtype=float)
    T = np.asarray(T, dtype=float)
    Y = len(N)
    if Y == 0:
        return np.array([], dtype=int)

    c0 = _neg_binom_ll(N, T, p0)
    c1 = _neg_binom_ll(N, T, p1)

    # dp[s] = min cost ending in state s at the current year
    dp0 = c0[0]
    dp1 = c1[0]
    # backpointers: bp[s][y] = previous state (0 or 1)
    bp = np.zeros((2, Y), dtype=np.int8)

    for y in range(1, Y):
        # to state 0
        from0 = dp0 + c0[y]
        from1 = dp1 + gamma + c0[y]
        if from0 <= from1:
            bp[0, y] = 0
            new_dp0 = from0
        else:
            bp[0, y] = 1
            new_dp0 = from1
        # to state 1
        from0_ = dp0 + gamma + c1[y]
        from1_ = dp1 + c1[y]
        if from0_ <= from1_:
            bp[1, y] = 0
            new_dp1 = from0_
        else:
            bp[1, y] = 1
            new_dp1 = from1_
        dp0, dp1 = new_dp0, new_dp1

    last_state = 0 if dp0 <= dp1 else 1
    states = np.zeros(Y, dtype=int)
    states[Y - 1] = last_state
    s = last_state
    for y in range(Y - 1, 0, -1):
        s = int(bp[s, y])
        states[y - 1] = s
    return states


def burst_intervals(states: np.ndarray, years: np.ndarray) -> List[Tuple[int, int]]:
    """Maximal contiguous runs of the burst state (state == 1).

    Returns ``(start_year, end_year)`` tuples in chronological order.
    """
    states = np.asarray(states, dtype=int)
    years = np.asarray(years, dtype=int)
    intervals: List[Tuple[int, int]] = []
    in_burst = False
    start = -1
    for i in range(len(states)):
        if states[i] == 1 and not in_burst:
            in_burst = True
            start = int(years[i])
        elif states[i] == 0 and in_burst:
            in_burst = False
            intervals.append((start, int(years[i - 1])))
    if in_burst:
        intervals.append((start, int(years[-1])))
    return intervals


def burst_strength(
    N: np.ndarray,
    T: np.ndarray,
    years: np.ndarray,
    p0: float,
    p1: float,
    interval: Tuple[int, int],
) -> float:
    """Log-likelihood-ratio of burst over baseline, summed over ``interval``.

    ``interval`` is ``(start_year, end_year)`` (inclusive).
    """
    mask = (years >= interval[0]) & (years <= interval[1])
    N_ = N[mask].astype(float)
    T_ = T[mask].astype(float)
    p0c = float(np.clip(p0, _LOG_FLOOR, 1.0 - _LOG_FLOOR))
    p1c = float(np.clip(p1, _LOG_FLOOR, 1.0 - _LOG_FLOOR))
    # c0 - c1 = N*ln(p1/p0) + (T-N)*ln((1-p1)/(1-p0))
    a = np.where(N_ > 0, N_ * np.log(p1c / p0c), 0.0)
    b = np.where((T_ - N_) > 0, (T_ - N_) * np.log((1.0 - p1c) / (1.0 - p0c)), 0.0)
    return float((a + b).sum())


def detect_recent_burst(
    years: np.ndarray,
    N: np.ndarray,
    T: np.ndarray,
    gamma: float = 1.0,
) -> dict:
    """Fit the two-state automaton for one topic and report its terminal burst.

    Returns a dict with keys::

        p0, p1, states, intervals,
        terminal_burst (start, end) or None,
        burst_strength (float or None),
        active_in_terminal_year (bool).

    ``terminal_burst`` is the burst interval (if any) that ends in the last year
    of the series — the "still active at the sequence end" signal used by the
    candidate-emerging filter.
    """
    years = np.asarray(years, dtype=int)
    N = np.asarray(N, dtype=float)
    T = np.asarray(T, dtype=float)

    p0, p1 = estimate_rates(N, T)
    if p1 <= p0 or p1 <= 0:
        return {
            "p0": p0, "p1": p1, "states": np.zeros(len(N), dtype=int),
            "intervals": [], "terminal_burst": None,
            "burst_strength": None, "active_in_terminal_year": False,
        }

    states = viterbi_path(N, T, p0, p1, gamma)
    intervals = burst_intervals(states, years)

    terminal = None
    strength = None
    active = False
    if intervals and intervals[-1][1] == int(years[-1]):
        terminal = intervals[-1]
        strength = burst_strength(N, T, years, p0, p1, terminal)
        active = True

    return {
        "p0": float(p0), "p1": float(p1), "states": states,
        "intervals": intervals, "terminal_burst": terminal,
        "burst_strength": strength, "active_in_terminal_year": active,
    }


def gamma_sensitivity(
    years: np.ndarray,
    N: np.ndarray,
    T: np.ndarray,
    gamma_grid: Tuple[float, ...] = (0.5, 1.0, 2.0),
) -> dict:
    """Burst detection under a grid of transition costs (Appendix S1 evidence)."""
    out = {}
    for g in gamma_grid:
        r = detect_recent_burst(years, N, T, gamma=g)
        out[str(g)] = {
            "terminal_burst": r["terminal_burst"],
            "burst_strength": r["burst_strength"],
            "active_in_terminal_year": r["active_in_terminal_year"],
        }
    return out


# ---------------------------------------------------------------------------
# Self-test (synthetic series)
# ---------------------------------------------------------------------------

def _self_test() -> None:
    years = np.arange(1970, 2026, dtype=int)  # 56 complete years

    # Case 1: cold-start -> terminal burst (should detect 2021-2025).
    T = np.full(len(years), 100.0)
    N = np.zeros(len(years))
    N[: -5] = 0.0
    N[-5:] = 20.0  # burst share 0.2 over 2021-2025
    r1 = detect_recent_burst(years, N, T, gamma=1.0)
    assert r1["active_in_terminal_year"] is True, r1
    assert r1["terminal_burst"] == (2021, 2025), r1["terminal_burst"]
    assert r1["burst_strength"] > 0, r1["burst_strength"]
    assert r1["p1"] > r1["p0"], (r1["p0"], r1["p1"])

    # Case 2: constant share everywhere -> no burst (baseline everywhere).
    N2 = np.full(len(years), 5.0)
    r2 = detect_recent_burst(years, N2, T, gamma=1.0)
    assert r2["active_in_terminal_year"] is False, r2

    # Case 3: burst that ended in 2020 (not active at terminal year).
    N3 = np.zeros(len(years))
    N3[10:15] = 30.0  # 1980-1984 burst
    r3 = detect_recent_burst(years, N3, T, gamma=1.0)
    assert r3["active_in_terminal_year"] is False, r3

    # Case 4: gamma sensitivity is consistent.
    sens = gamma_sensitivity(years, N, T)
    assert set(sens.keys()) == {"0.5", "1.0", "2.0"}, sens

    print("Kleinberg-Binomial self-test PASSED")
    print(f"  case1 p0={r1['p0']:.4f} p1={r1['p1']:.4f} "
          f"strength={r1['burst_strength']:.2f} terminal={r1['terminal_burst']}")


if __name__ == "__main__":
    _self_test()
