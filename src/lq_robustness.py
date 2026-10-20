"""P0-3: Statistical defense of the Location Quotient (LQ) specialization claims.

The manuscript reports the signature niche of seven resource-rich countries:
the seven with the largest positive RRMI among the countries that are both
above the N_c >= 10 measurability threshold and hold at least 1,000 km^2 of
mangrove. That country set is re-derived from the artifacts on every run
(_resolve_niches) and asserted against the frozen list below, so a change in
country extraction, the topic model or the RRMI table raises instead of
silently renaming countries. Two threats to the claim are (i)
threshold-survivor inflation: a small denominator N_c trivially produces
LQ > 1; and (ii) sampling noise: the observed LQ may be indistinguishable
from a random assignment of topics.

This module supplies the two robustness checks cited in the manuscript:

1. **Permutation null model** — topic labels are permuted across documents
   (preserving each country's document total N_c and each topic's document
   total N_k), recomputing the fractional N_{c,k} and LQ for the seven niches,
   1,000 replicates. The one-sided p-value is the fraction of replicates whose
   permuted LQ meets or exceeds the observed value.

2. **Empirical-Bayes (beta-binomial) shrinkage** — each country--topic
   proportion p_{c,k} = N_{c,k}/N_c is shrunk toward the global topic share
   p_k = N_k/N under a Beta(alpha_k, beta_k) prior fitted by moment matching,
   giving a posterior LQ that shrinks toward 1. A niche "survives" shrinkage if
   its posterior LQ remains > 1.

Usage:
    python src/lq_robustness.py              # default config.yaml
    python src/lq_robustness.py --config path/to/config.yaml
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import Config, get_config
from src.spatial_topic_analysis import (
    _COUNTRY_NAME_MAPPING_DEFAULT,
    _CU_FALSE_POSITIVES,
    fractional_country_topic,
    load_topic_labels,
    merge_country_topic,
)
from src.utils import Timer, canonical_countries, ensure_dir

logger = logging.getLogger(__name__)

# Reporting threshold: LQ is defined only for N_c >= 10 *topic-assigned*
# fractional documents. Documents left unassigned by BERTopic are excluded,
# so this N_c is smaller than the full-corpus fractional count behind the
# RRMI (see the manuscript's counting-convention table).
_N_C_MIN = 10

# "Resource-rich" boundary shared with the bivariate map and the niche rule.
_RESOURCE_RICH_MIN_AREA_KM2 = 1000.0

# Number of signature niches reported in the manuscript (Table tab:lq).
_N_NICHES = 7

# The signature niches reported in the manuscript (Table tab:lq). The COUNTRY
# SET is not taken on trust: _resolve_niches() re-derives it on every run as
# the _N_NICHES largest positive RRMI values among the countries that are both
# measurable (N_c >= _N_C_MIN) and resource-rich
# (>= _RESOURCE_RICH_MIN_AREA_KM2 of mangrove), and raises if the derived set
# differs from the frozen list below -- the same fail-loudly pattern as
# src/topic_clusters.py. The TOPIC per country is the reported choice: the
# country's highest-LQ topic for six of the seven, and T40 (LQ 17.2) rather
# than the numerically higher T51 (LQ 19.9) for Colombia, the coherent
# national line of work (see the manuscript). (country, topic_id)
_NICHES: List[Tuple[str, int]] = [
    ("Colombia", 40),
    ("Indonesia", 53),
    ("Malaysia", 18),
    ("Bangladesh", 38),
    ("Mexico", 70),
    ("Nigeria", 50),
    ("Brazil", 15),
]


# Permutation replicates.
_N_PERMS = 1000


def _unfold_country_names(mapping: Dict[str, str]) -> Dict[str, str]:
    """Reverse the country fold (analysis name -> canonical CU spelling).

    ``area_pub_deficit.csv`` is written in the canonical ``CU`` spelling while
    the fractional country-topic matrix uses the folded analysis names, so the
    two are joined through this map. Where several canonical names fold onto the
    same analysis name the first one wins; the folded spellings involved here
    are preprocessing typos (e.g. ``Venzuela``), so nothing published collides.
    """
    unfold: Dict[str, str] = {}
    for canonical, folded in mapping.items():
        unfold.setdefault(folded, canonical)
    return unfold


def _derive_niche_countries(
    config: Config,
    fractional_df: pd.DataFrame,
    unfold: Dict[str, str],
) -> List[str]:
    """Re-derive the reported niche countries from the current artifacts.

    Rule (manuscript Sec. si): of the countries that are both measurable
    (N_c >= _N_C_MIN topic-assigned fractional documents) and resource-rich
    (mangrove area >= _RESOURCE_RICH_MIN_AREA_KM2), take the _N_NICHES with the
    largest positive RRMI.

    Args:
        config: Loaded configuration.
        fractional_df: Long-form ``Country, Topic, N_ck`` frame.
        unfold: Reverse country fold from :func:`_unfold_country_names`.

    Returns:
        Country names in descending RRMI order.

    Raises:
        FileNotFoundError: If ``area_pub_deficit.csv`` has not been generated.
    """
    deficit_csv = config.paths.processed_dir / "area_pub_deficit.csv"
    if not deficit_csv.exists():
        raise FileNotFoundError(
            f"{deficit_csv} not found: run `python -m src.area_pub_deficit` first."
        )

    deficit = pd.read_csv(deficit_csv, encoding="utf-8-sig")
    if "country" in deficit.columns:
        deficit = deficit.rename(columns={"country": "Country"})

    n_c = fractional_df.groupby("Country")["N_ck"].sum()
    n_c.index = n_c.index.map(lambda c: unfold.get(c, c))
    deficit["N_c"] = deficit["Country"].map(n_c).fillna(0.0)

    eligible = deficit[
        (deficit["N_c"] >= _N_C_MIN)
        & (deficit["area"] >= _RESOURCE_RICH_MIN_AREA_KM2)
        & (deficit["deficit"] > 0.0)
    ].sort_values("deficit", ascending=False)

    derived = eligible["Country"].head(_N_NICHES).tolist()
    logger.info(
        "Derived niche countries (measurable AND >= %.0f km2, largest "
        "positive RRMI): %s",
        _RESOURCE_RICH_MIN_AREA_KM2,
        derived,
    )
    return derived


def _resolve_niches(
    config: Config,
    fractional_df: pd.DataFrame,
    unfold: Dict[str, str],
) -> Tuple[List[Tuple[str, int]], List[str]]:
    """Derive the country set, assert it equals the frozen list, return niches.

    Returns:
        (niches, derived_countries).

    Raises:
        ValueError: If the derived set differs from ``_NICHES``.
    """
    derived = _derive_niche_countries(config, fractional_df, unfold)
    frozen = sorted(c for c, _ in _NICHES)
    if sorted(derived) != frozen:
        raise ValueError(
            "Derived signature-niche countries do not match the frozen list in "
            f"_NICHES. derived={sorted(derived)} frozen={frozen}. The country "
            "extraction, the topic model or the RRMI table changed; re-check the "
            "manuscript's niche table instead of renaming countries silently."
        )
    return list(_NICHES), derived


def _build_doc_level(
    merged: pd.DataFrame,
    mapping: Dict[str, str],
    niche_countries: List[str],
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, float]]:
    """Build doc-level topic array and per-country fractional-weight vectors.

    Returns:
        (topic_arr, weight_by_country, n_c_by_country) where ``topic_arr[d]`` is
        the topic of document d, ``weight_by_country[c][d]`` is country c's
        fractional weight on document d (0 if absent), and ``n_c_by_country[c]``
        is country c's total fractional document count.
    """
    topic_list: List[int] = []
    doc_weights: List[List[Tuple[str, float]]] = []

    for _, row in merged.iterrows():
        cu_val = str(row["CU"]) if pd.notna(row["CU"]) else "Unknown"
        # Fold and de-duplicate: a "China;Taiwan" record contributes ONE country
        # at weight 1.0 (see src.utils.canonical_countries).
        countries = canonical_countries(
            cu_val, fold=mapping, drop=_CU_FALSE_POSITIVES
        )
        if not countries:
            continue
        w = 1.0 / len(countries)
        topic_list.append(int(row["Topic"]))
        doc_weights.append([(c, w) for c in countries])

    topic_arr = np.asarray(topic_list, dtype=int)
    n_docs = len(topic_arr)

    weight_by_country: Dict[str, np.ndarray] = {c: np.zeros(n_docs) for c in niche_countries}
    for d, cw_list in enumerate(doc_weights):
        for c, w in cw_list:
            if c in weight_by_country:
                weight_by_country[c][d] = w

    n_c_by_country = {c: float(wv.sum()) for c, wv in weight_by_country.items()}
    return topic_arr, weight_by_country, n_c_by_country


def _observed_lq(
    topic_arr: np.ndarray,
    weight_by_country: Dict[str, np.ndarray],
    n_c_by_country: Dict[str, float],
    niches: List[Tuple[str, int]],
) -> Dict[Tuple[str, int], float]:
    """Observed LQ for each niche from the un-permuted assignment."""
    n_docs = len(topic_arr)
    lq = {}
    for c, k in niches:
        n_ck = float(weight_by_country[c][topic_arr == k].sum())
        n_c = n_c_by_country[c]
        n_k = float((topic_arr == k).sum())
        lq[(c, k)] = (n_ck / n_c) / (n_k / n_docs) if (n_c > 0 and n_k > 0) else 0.0
    return lq


def _permutation_test(
    topic_arr: np.ndarray,
    weight_by_country: Dict[str, np.ndarray],
    n_c_by_country: Dict[str, float],
    observed_lq: Dict[Tuple[str, int], float],
    niches: List[Tuple[str, int]],
    seed: int = 42,
) -> Dict[Tuple[str, int], float]:
    """One-sided permutation p-values for each niche.

    Topic labels are permuted across documents; both country totals N_c and
    topic totals N_k are preserved exactly because each document's country set
    (hence fractional weight) is held fixed and only its topic is reassigned.
    """
    rng = np.random.default_rng(seed)
    n_docs = len(topic_arr)
    p_values: Dict[Tuple[str, int], float] = {}

    for c, k in niches:
        w_c = weight_by_country[c]
        n_c = n_c_by_country[c]
        n_k = float((topic_arr == k).sum())
        obs = observed_lq[(c, k)]
        geq = 0
        for _ in range(_N_PERMS):
            shuffled = topic_arr[rng.permutation(n_docs)]
            n_ck_perm = float(w_c[shuffled == k].sum())
            lq_perm = (n_ck_perm / n_c) / (n_k / n_docs)
            if lq_perm >= obs:
                geq += 1
        p_values[(c, k)] = (1 + geq) / (1 + _N_PERMS)

    return p_values


def _beta_binomial_shrinkage(
    fractional_df: pd.DataFrame,
    niches: List[Tuple[str, int]],
) -> Tuple[pd.DataFrame, Dict[Tuple[str, int], float]]:
    """Empirical-Bayes beta-binomial shrinkage of each country--topic proportion.

    Returns:
        (shrunk_lq_matrix, shrunk_lq_by_niche). The matrix holds posterior LQ
        for all measurable (country, topic) pairs; the dict extracts the seven
        reported niches.
    """
    n_ck = fractional_df.pivot_table(
        index="Country", columns="Topic", values="N_ck", fill_value=0.0
    )
    n_ck = n_ck.reindex(sorted(n_ck.columns), axis=1)

    n_c = n_ck.sum(axis=1)
    n_k = n_ck.sum(axis=0)
    n_total = float(n_ck.values.sum())

    # Restrict to the measurable set (N_c >= threshold).
    measurable = n_ck.loc[n_c >= _N_C_MIN].copy()
    n_c_m = n_c.loc[measurable.index]
    p_k = n_k / n_total

    shrunk = pd.DataFrame(np.nan, index=measurable.index, columns=measurable.columns)

    for k in measurable.columns:
        pk = float(p_k[k])
        p_hat = measurable[k] / n_c_m  # observed proportions across countries
        # Weighted variance of p_hat around the global prior mean p_k.
        w = n_c_m.values
        var = float(np.average((p_hat.values - pk) ** 2, weights=w))
        # Moment-matched beta prior; fall back to uniform when variance is
        # degenerate (zero) or exceeds the beta upper bound p_k(1-p_k).
        bound = pk * (1.0 - pk)
        if var > 0.0 and var < bound:
            nu = pk * (1.0 - pk) / var - 1.0
            alpha_k = pk * nu
            beta_k = (1.0 - pk) * nu
        else:
            alpha_k, beta_k = 1.0, 1.0  # weakly informative uniform prior
        # Posterior mean of p_{c,k}.
        posterior = (measurable[k].values + alpha_k) / (n_c_m.values + alpha_k + beta_k)
        shrunk[k] = posterior / pk if pk > 0 else np.nan

    shrunk_lq_by_niche: Dict[Tuple[str, int], float] = {}
    for c, k in niches:
        if c in shrunk.index and k in shrunk.columns:
            val = shrunk.loc[c, k]
            shrunk_lq_by_niche[(c, k)] = float(val) if pd.notna(val) else float("nan")
        else:
            shrunk_lq_by_niche[(c, k)] = float("nan")

    return shrunk, shrunk_lq_by_niche


def run_lq_robustness(config: Config) -> Dict:
    """Run both robustness checks and write CSV / JSON / report outputs."""
    paths = config.paths

    mapping = {
        **_COUNTRY_NAME_MAPPING_DEFAULT,
        **config.country_analysis.country_name_mapping,
    }

    with Timer("Merging country-topic data"):
        merged = merge_country_topic(config)

    with Timer("Fractional counting"):
        fractional_df = fractional_country_topic(merged, mapping=mapping)

    unfold = _unfold_country_names(mapping)
    niches, derived_countries = _resolve_niches(config, fractional_df, unfold)

    topic_labels = load_topic_labels(config)
    niche_countries = sorted({c for c, _ in niches})

    with Timer("Building doc-level weight vectors"):
        topic_arr, weight_by_country, n_c_by_country = _build_doc_level(
            merged, mapping, niche_countries
        )

    observed_lq = _observed_lq(
        topic_arr, weight_by_country, n_c_by_country, niches
    )

    with Timer(f"Permutation null ({_N_PERMS} replicates)"):
        p_values = _permutation_test(
            topic_arr, weight_by_country, n_c_by_country, observed_lq, niches
        )

    with Timer("Beta-binomial shrinkage"):
        shrunk_matrix, shrunk_lq = _beta_binomial_shrinkage(fractional_df, niches)

    # Assemble results.
    rows = []
    for c, k in niches:
        n_c = n_c_by_country[c]
        n_ck = float(weight_by_country[c][topic_arr == k].sum())
        rows.append({
            "Country": c,
            "Topic": k,
            "Topic_Label": topic_labels.get(k, f"T{k}"),
            "N_c": round(n_c, 2),
            "N_ck": round(n_ck, 2),
            "observed_LQ": round(observed_lq[(c, k)], 3),
            "perm_p_value": round(p_values[(c, k)], 4),
            "shrunk_LQ": round(shrunk_lq[(c, k)], 3),
            "survives_permutation": bool(p_values[(c, k)] < 0.05),
            "survives_shrinkage": bool(shrunk_lq[(c, k)] > 1.0),
        })

    results_df = pd.DataFrame(rows)

    summary = {
        "n_permutations": _N_PERMS,
        "n_c_min": _N_C_MIN,
        "niche_selection_rule": (
            f"top-{_N_NICHES} positive RRMI among countries with "
            f"N_c >= {_N_C_MIN} topic-assigned fractional documents and "
            f">= {_RESOURCE_RICH_MIN_AREA_KM2:.0f} km2 of mangrove"
        ),
        "niche_countries_derived": derived_countries,
        "niches": rows,
        "all_survive_permutation": bool((results_df["perm_p_value"] < 0.05).all()),
        "all_survive_shrinkage": bool((results_df["shrunk_LQ"] > 1.0).all()),
        "max_perm_p_value": float(results_df["perm_p_value"].max()),
        "min_shrunk_LQ": float(results_df["shrunk_LQ"].min()),
    }

    csv_path = paths.processed_dir / "lq_robustness.csv"
    json_path = paths.processed_dir / "lq_robustness_summary.json"
    report_path = paths.processed_dir / "lq_robustness_report.md"

    results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # Human-readable report.
    lines = [
        "# LQ Robustness (P0-3)",
        "",
        f"- Permutation null: {_N_PERMS} replicates, one-sided (permuted LQ >= observed).",
        f"- Measurability threshold: N_c >= {_N_C_MIN} topic-assigned fractional "
        "documents (unassigned documents excluded).",
        f"- Niche selection: {summary['niche_selection_rule']}.",
        f"- Countries re-derived from the current artifacts: "
        f"{', '.join(derived_countries)}.",
        "- Beta-binomial shrinkage: empirical-Bayes posterior LQ shrunk toward 1.",
        "",
        "| Country | Topic | Label | N_c | N_ck | LQ_obs | p_perm | LQ_shrunk | survives |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        surv = "yes" if (r["survives_permutation"] and r["survives_shrinkage"]) else "no"
        lines.append(
            f"| {r['Country']} | T{r['Topic']} | {r['Topic_Label']} | {r['N_c']:.1f} | "
            f"{r['N_ck']:.1f} | {r['observed_LQ']:.2f} | {r['perm_p_value']:.4f} | "
            f"{r['shrunk_LQ']:.2f} | {surv} |"
        )
    lines += [
        "",
        f"All {len(niches)} niches survive the permutation null at p < 0.05: "
        f"{summary['all_survive_permutation']}.",
        f"All {len(niches)} niches retain shrunk LQ > 1: "
        f"{summary['all_survive_shrinkage']} "
        f"(min shrunk LQ = {summary['min_shrunk_LQ']:.2f}).",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    logger.info("LQ robustness CSV: %s", csv_path)
    logger.info("LQ robustness summary: %s", json_path)
    logger.info("LQ robustness report: %s", report_path)
    return summary


def main() -> None:
    import argparse

    from src.utils import setup_logging

    parser = argparse.ArgumentParser(description="LQ robustness (P0-3): permutation null + shrinkage")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = get_config(args.config)
    setup_logging(config.system.log_level, config.system.log_file)

    try:
        run_lq_robustness(config)
    except Exception:
        logger.exception("LQ robustness failed")
        raise


if __name__ == "__main__":
    main()
