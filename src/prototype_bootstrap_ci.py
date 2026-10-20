"""#3-C: Year-stratified bootstrap 95% CI for candidate burst strengths.

For each candidate emerging research front (a topic passing the Kleinberg
three-criterion filter), we bootstrap its *burst strength* — the
log-likelihood-ratio of the burst state over the baseline state, summed across
the detected terminal burst interval (see ``src/burst_detection_kleinberg.py``).

Unlike the earlier global-document resample, the resampling is **stratified by
publication year**: within each year we draw documents with replacement, so the
yearly corpus totals ``T_y`` are held fixed and only the topic-count noise is
perturbed.  This keeps the normalization base (the yearly denominator) intact,
so the CI reflects the uncertainty in the burst signal rather than a composite
of topic-assignment and corpus-size noise.

Each resample **re-runs the whole automaton**: ``detect_recent_burst`` re-estimates
``p0`` and ``p1`` by hard-EM on the resampled counts and then re-decodes the
Viterbi path at ``gamma``.  Holding the rates at their observed values and only
re-decoding would understate the uncertainty, because the rate estimates are
themselves the noisiest part of a cold-start topic's signal.

Detection criterion (aligned with the candidate definition)
-----------------------------------------------------------
A replicate counts as a **detection** only when its terminal burst satisfies the
*full* recency criterion of ``src.emerging_tech_detection`` — the terminal burst
ends in ``automaton_end``, begins no earlier than ``MIN_BURST_ONSET``, and spans
at least ``MIN_BURST_LENGTH`` years.  This makes the bootstrap estimand "the
probability that the topic would be re-labelled a candidate front", which is
what the manuscript reports.  Two weaker bookkeeping columns are retained purely
for audit:

``pct_no_terminal_burst``
    fraction of replicates with no burst interval active in the final year
    (the earlier, looser criterion — an interval that began in, say, the 1990s
    still counted as "detected", so it inflated apparent stability);
``pct_active_but_recency_failed``
    fraction where a terminal burst was active but failed the onset or span
    clause.

The reported 95% interval is computed over the *detection* replicates only, so
the interval and the loss fraction describe the same event.

Usage:
    python src/prototype_bootstrap_ci.py              # default config.yaml
    python src/prototype_bootstrap_ci.py --config path/to/config.yaml
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from src.burst_detection_kleinberg import detect_recent_burst
from src.config import Config, get_config
from src.emerging_tech_detection import (
    AUTOMATON_END,
    GAMMA,
    MIN_BURST_LENGTH,
    MIN_BURST_ONSET,
    compute_yearly_topic_counts,
    detect_emerging_tech_prototypes,
)
from src.utils import Timer

logger = logging.getLogger(__name__)

_N_BOOT = 1000


def _short_label(keywords: str, max_words: int = 3) -> str:
    """Abbreviate a c-TF-IDF keyword string to a short display label."""
    if not keywords or keywords == "nan":
        return "---"
    words = keywords.replace("_", " ").split(", ")
    return ", ".join(words[:max_words])


def _topic_series(
    yearly_counts: pd.DataFrame,
    total_per_year: dict,
    topic_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ``(years, N, T)`` over the corpus-covered years up to ``AUTOMATON_END``.

    Only years present in ``yearly_counts`` enter the series, so there is no
    ``T_y = 0`` bin; ``burst_intervals`` is therefore contiguous in the supplied
    year sequence (see ``src/burst_detection_kleinberg``).
    """
    all_years = sorted(int(y) for y in yearly_counts.index)
    complete_years = [y for y in all_years if y <= AUTOMATON_END]
    T = np.array([total_per_year.get(y, 0) for y in complete_years], dtype=float)
    N = np.array(
        [yearly_counts.loc[y, topic_id] if topic_id in yearly_counts.columns else 0
         for y in complete_years],
        dtype=float,
    )
    return np.array(complete_years, dtype=int), N, T


def _recency_qualified(terminal: tuple[int, int] | None, active: bool) -> bool:
    """Does the terminal burst satisfy all three recency clauses?"""
    if terminal is None or not active:
        return False
    onset_ok = terminal[0] >= MIN_BURST_ONSET
    span_ok = (terminal[1] - terminal[0] + 1) >= MIN_BURST_LENGTH
    return bool(onset_ok and span_ok)


def _topic_terminal_outcome(
    yearly_counts: pd.DataFrame,
    total_per_year: dict,
    topic_id: int,
    gamma: float = GAMMA,
) -> tuple[float | None, str, bool, bool]:
    """Re-run the two-state automaton for one topic.

    Returns ``(strength, interval_str, active_in_terminal_year, recency_ok)``.
    ``strength`` is None when no terminal burst exists in the series;
    ``recency_ok`` additionally applies the onset and span clauses.
    """
    years, N, T = _topic_series(yearly_counts, total_per_year, topic_id)
    det = detect_recent_burst(years, N, T, gamma=gamma)
    interval = ""
    if det["terminal_burst"] is not None:
        interval = f"{det['terminal_burst'][0]}-{det['terminal_burst'][1]}"
    active = bool(det["active_in_terminal_year"])
    return (
        det["burst_strength"],
        interval,
        active,
        _recency_qualified(det["terminal_burst"], active),
    )


def _topic_terminal_strength(
    yearly_counts: pd.DataFrame,
    total_per_year: dict,
    topic_id: int,
    gamma: float = GAMMA,
) -> tuple[float | None, str, bool]:
    """Backwards-compatible wrapper: ``(strength, interval, active)``."""
    strength, interval, active, _ = _topic_terminal_outcome(
        yearly_counts, total_per_year, topic_id, gamma=gamma)
    return strength, interval, active


def run_bootstrap_ci(config: Config) -> pd.DataFrame:
    paths = config.paths

    doc_topic = pd.read_csv(
        paths.processed_dir / config.topic_modeling.output.doc_topic_dist,
        encoding="utf-8-sig",
        usecols=["Topic", "PY"],
    )
    doc_topic = doc_topic.dropna(subset=["PY", "Topic"])
    doc_topic["Topic"] = doc_topic["Topic"].astype(int)
    doc_topic["PY"] = pd.to_numeric(doc_topic["PY"], errors="coerce").astype("Int64")
    doc_topic = doc_topic.dropna(subset=["PY"])

    # --- observed candidates (from the Kleinberg three-criterion filter) ----
    yearly_counts, total_per_year = compute_yearly_topic_counts(doc_topic)
    df_emerging = detect_emerging_tech_prototypes(yearly_counts, total_per_year)
    if df_emerging.empty:
        logger.warning("[WARN] 无候选新兴前沿，跳过 bootstrap CI。")
        return pd.DataFrame()

    candidates = [
        (int(r.Topic), _short_label(str(getattr(r, "Top_Keywords", ""))))
        for r in df_emerging.itertuples(index=False)
    ]
    observed = {
        tid: _topic_terminal_strength(yearly_counts, total_per_year, tid)[0]
        for tid, _ in candidates
    }

    # --- year-stratified bootstrap (T_y fixed) -----------------------------
    rng = np.random.default_rng(42)
    boot_samples: dict[int, list[float]] = {tid: [] for tid, _ in candidates}
    n_no_active: dict[int, int] = {tid: 0 for tid, _ in candidates}
    n_no_recency: dict[int, int] = {tid: 0 for tid, _ in candidates}
    year_groups = [(y, g) for y, g in doc_topic.groupby("PY")]

    with Timer(f"Year-stratified bootstrap ({_N_BOOT} resamples)"):
        for _ in range(_N_BOOT):
            parts = []
            for _year, grp in year_groups:
                idx = rng.integers(0, len(grp), size=len(grp))
                parts.append(grp.iloc[idx])
            resampled = pd.concat(parts)
            yc, tp = compute_yearly_topic_counts(resampled)
            for tid, _ in candidates:
                s, _iv, active, recency_ok = _topic_terminal_outcome(yc, tp, tid)
                has_strength = s is not None and not np.isnan(s)
                if has_strength and recency_ok:
                    boot_samples[tid].append(s)
                else:
                    n_no_recency[tid] += 1
                    if not (has_strength and active):
                        n_no_active[tid] += 1

    rows = []
    for tid, label in candidates:
        samples = np.array(boot_samples[tid])
        if len(samples) == 0:
            lo = hi = float("nan")
        else:
            lo, hi = np.percentile(samples, [2.5, 97.5])
        n_act = n_no_active[tid]
        n_rec = n_no_recency[tid]
        rows.append({
            "Topic": tid,
            "Label": label,
            "Burst_Strength": round(observed[tid], 2) if observed[tid] is not None else float("nan"),
            "CI_low": round(float(lo), 2),
            "CI_high": round(float(hi), 2),
            "excludes_0": bool(lo > 0.0),
            "n_detected": int(len(samples)),
            "pct_detected": round(100.0 * len(samples) / _N_BOOT, 1),
            "pct_no_recency_terminal_burst": round(100.0 * n_rec / _N_BOOT, 1),
            "pct_active_but_recency_failed": round(100.0 * (n_rec - n_act) / _N_BOOT, 1),
            "pct_no_terminal_burst": round(100.0 * n_act / _N_BOOT, 1),
        })

    results = pd.DataFrame(rows)
    all_exclude_0 = bool((results["excludes_0"] == True).all()) if not results.empty else False  # noqa: E712

    csv_path = paths.processed_dir / "prototype_bootstrap_ci.csv"
    json_path = paths.processed_dir / "prototype_bootstrap_ci_summary.json"
    results.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(
        json.dumps({
            "n_bootstrap": _N_BOOT,
            "bootstrap_scheme": "year-stratified (T_y fixed, topic counts resampled); rates re-estimated by hard-EM and path re-decoded on every replicate",
            "detection_criterion": (
                "terminal burst ends in automaton_end AND begins >= "
                f"{MIN_BURST_ONSET} AND spans >= {MIN_BURST_LENGTH} years "
                "(full recency criterion)"
            ),
            "weaker_audit_criterion": "no burst interval active in the final year",
            "all_CIs_exclude_0": all_exclude_0,
            "candidates": rows,
        }, indent=2),
        encoding="utf-8",
    )

    logger.info("Prototype burst-strength bootstrap CI CSV: %s", csv_path)
    for r in rows:
        logger.info(
            "  T%d %-32s W=%7.2f  CI=[%7.2f, %7.2f]  detected=%.1f%%  "
            "no_recency=%.1f%%  (loose no_active=%.1f%%)",
            r["Topic"], r["Label"], r["Burst_Strength"],
            r["CI_low"], r["CI_high"], r["pct_detected"],
            r["pct_no_recency_terminal_burst"], r["pct_no_terminal_burst"],
        )
    logger.info("All CIs exclude 0: %s", all_exclude_0)
    return results


def main() -> None:
    import argparse

    from src.utils import setup_logging

    parser = argparse.ArgumentParser(description="Candidate burst-strength bootstrap CI (#3-C)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = get_config(args.config)
    setup_logging(config.system.log_level, config.system.log_file)
    try:
        run_bootstrap_ci(config)
    except Exception:
        logger.exception("Bootstrap CI failed")
        raise


if __name__ == "__main__":
    main()
