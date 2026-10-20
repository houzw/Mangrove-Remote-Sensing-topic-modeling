"""Verify the P2/P3 claims of manuscript-v3.tex Sec. si (National Thematic
Specialization) against the current pipeline artifacts.

Checks:
  1. How many countries are measurable (N_c >= 10 topic-assigned fractional pubs)?
     How many of them have LQ > 1 in exactly one topic? Who are they?
  2. Countries with N_c < 10: how many? how many are "resource-rich"
     (mangrove area >= 1000 km^2)? their area share / pub share.
  3. Overlap of the N_c < 10 group with the negative-RRMI (over-publishing) group
     and with the ten most negative RRMI countries.
  4. Where do the seven Table tab:lq countries sit (N_c, RRMI)?
  5. T50 / T54 canonical labels + cluster + top c-TF-IDF terms.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_config  # noqa: E402
from src.lq_robustness import _N_C_MIN, _NICHES  # noqa: E402
from src.spatial_topic_analysis import (  # noqa: E402
    _COUNTRY_NAME_MAPPING_DEFAULT,
    fractional_country_topic,
    load_topic_labels,
    merge_country_topic,
)

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)


def main() -> None:
    config = get_config()
    paths = config.paths

    mapping = {
        **_COUNTRY_NAME_MAPPING_DEFAULT,
        **config.country_analysis.country_name_mapping,
    }

    merged = merge_country_topic(config)
    frac = fractional_country_topic(merged, mapping=mapping)
    n_ck = frac.pivot_table(
        index="Country", columns="Topic", values="N_ck", fill_value=0.0
    ).reindex(sorted(frac["Topic"].unique()), axis=1)
    n_c = n_ck.sum(axis=1)
    n_k = n_ck.sum(axis=0)
    n_total = float(n_ck.values.sum())

    lq = (n_ck.div(n_c, axis=0)).div(n_k / n_total, axis=1)

    print("=" * 78)
    print("0. GLOBAL")
    print("=" * 78)
    print(f"topic-assigned documents (with country info): {len(merged)}")
    print(f"countries present: {len(n_c)}   N (fractional docs) = {n_total:.1f}")
    print(f"manuscript / code measurability threshold N_c >= {_N_C_MIN}")
    print(f"config spatial_topic_analysis.min_docs_per_country = "
          f"{config.spatial_topic_analysis.min_docs_per_country}")

    measurable = n_c[n_c >= _N_C_MIN]
    below = n_c[n_c < _N_C_MIN]
    print(f"measurable countries: {len(measurable)}")
    print(f"below-threshold countries (N_c > 0): {len(below)}")

    print()
    print("=" * 78)
    print("1. MEASURABLE COUNTRIES: number of topics with LQ > 1")
    print("=" * 78)
    lq_m = lq.loc[measurable.index]
    n_over = (lq_m > 1.0).sum(axis=1)
    top = lq_m.idxmax(axis=1)
    topval = lq_m.max(axis=1)
    tbl = pd.DataFrame({
        "N_c": measurable.round(2),
        "n_topics_LQ>1": n_over,
        "maxLQ_topic": top,
        "maxLQ": topval.round(2),
    }).sort_values("N_c", ascending=False)
    print(tbl.to_string())
    print()
    print(f"countries with exactly one topic at LQ>1: {(n_over == 1).sum()}")
    print(f"countries with zero topics at LQ>1:       {(n_over == 0).sum()}")
    print(f"countries with >=2 topics at LQ>1:       {(n_over >= 2).sum()}")
    if (n_over >= 2).any():
        print("  ->", list(n_over[n_over >= 2].index))
    if (n_over == 0).any():
        print("  ->", list(n_over[n_over == 0].index))

    print()
    print("  Table tab:lq niches (hard-coded in src/lq_robustness._NICHES):")
    niche_countries = sorted({c for c, _ in _NICHES})
    niche_tbl = pd.DataFrame({
        "N_c": n_c.reindex(niche_countries).round(2),
        "reported_topic": [dict(_NICHES)[c] for c in niche_countries],
    })
    niche_tbl["LQ_of_reported_topic"] = [
        round(float(lq.loc[c, dict(_NICHES)[c]]), 3) for c in niche_countries
    ]
    niche_tbl["n_topics_LQ>1"] = n_over.reindex(niche_countries)
    print(niche_tbl.to_string())

    print()
    print("=" * 78)
    print("2. BELOW-THRESHOLD COUNTRIES (N_c < 10) + mangrove area")
    print("=" * 78)
    deficit = pd.read_csv(paths.processed_dir / "area_pub_deficit.csv")
    deficit = deficit.rename(columns={"country": "Country"})
    below_df = pd.DataFrame({"N_c": below.round(2)})
    below_df = below_df.merge(
        deficit[["Country", "area", "pubs", "area_share", "pub_share", "deficit"]],
        left_index=True, right_on="Country", how="left",
    ).set_index("Country")
    below_df = below_df.sort_values("area", ascending=False)
    print(below_df.to_string())
    print()
    in_deficit = below_df["area"].notna().sum()
    print(f"below-threshold countries with a mangrove-area row: {in_deficit}")
    rich = below_df[(below_df["area"] >= 1000)]
    print(f"below-threshold AND area >= 1000 km^2: {len(rich)}")
    print(rich.to_string())
    print()
    if len(rich):
        print(f"  sum area share = {rich['area_share'].sum():.4f} "
              f"({100 * rich['area_share'].sum():.1f}%)")
        print(f"  sum pub  share = {rich['pub_share'].sum():.4f} "
              f"({100 * rich['pub_share'].sum():.1f}%)")
    all_below_deficit = below_df[below_df["area"].notna()]
    if len(all_below_deficit):
        print(f"  ALL below-threshold with area: n={len(all_below_deficit)} "
              f"area_share={100 * all_below_deficit['area_share'].sum():.1f}% "
              f"pub_share={100 * all_below_deficit['pub_share'].sum():.1f}%")

    print()
    print("=" * 78)
    print("3. GROUP OVERLAP: N_c < 10  vs  RRMI < 0  vs  top-10 most negative RRMI")
    print("=" * 78)
    neg = set(deficit.loc[deficit["deficit"] < 0, "Country"])
    top10 = set(
        deficit.sort_values("deficit").head(10)["Country"]
    )
    below_set = set(below.index)
    print(f"negative-RRMI (over-publishing) countries: {len(neg)}")
    print(f"ten most negative: {sorted(top10)}")
    print(f"below-threshold set size: {len(below_set)}")
    print(f"below-threshold ∩ negative-RRMI: {len(below_set & neg)} -> "
          f"{sorted(below_set & neg)}")
    print(f"below-threshold ∩ ten-most-negative: {len(below_set & top10)} -> "
          f"{sorted(below_set & top10)}")
    print(f"below-threshold countries NOT in deficit table: "
          f"{sorted(c for c in below_set if c not in set(deficit['Country']))}")

    print()
    print("=" * 78)
    print("4. SEVEN TAB:lq COUNTRIES: N_c, RRMI, area")
    print("=" * 78)
    seven = pd.DataFrame(index=niche_countries)
    seven["N_c"] = n_c.reindex(niche_countries).round(2)
    seven = seven.merge(
        deficit.set_index("Country")[["area", "deficit", "rank"]],
        left_index=True, right_index=True, how="left",
    )
    seven["measurable"] = seven["N_c"] >= _N_C_MIN
    print(seven.to_string())

    print()
    print("=" * 78)
    print("5. TOPIC LABELS / CLUSTERS (T50, T54, and the seven niches)")
    print("=" * 78)
    labels = load_topic_labels(config)
    clusters = pd.read_csv(paths.processed_dir / "topic_clusters.csv")
    cl = clusters.set_index("Topic")["Cluster"].to_dict()
    clname = clusters.set_index("Topic")["Cluster_Name"].to_dict()
    info = pd.read_csv(paths.processed_dir / "topics_info.csv")
    info = info.set_index("Topic")
    rep = info["Representation"] if "Representation" in info.columns else None

    def top_terms(tid: int, n: int = 8) -> str:
        if rep is None or tid not in rep.index:
            return "n/a"
        val = rep.loc[tid]
        if isinstance(val, str):
            import ast
            try:
                val = ast.literal_eval(val)
            except Exception:
                return val[:120]
        return "; ".join(list(val)[:n])

    topics_of_interest = sorted({t for _, t in _NICHES} | {50, 54, 53})
    out = pd.DataFrame({
        "canonical_label": [labels.get(t, "?") for t in topics_of_interest],
        "cluster": [cl.get(t, "?") for t in topics_of_interest],
        "cluster_name": [clname.get(t, "?") for t in topics_of_interest],
        "top_c-TF-IDF_terms": [top_terms(t) for t in topics_of_interest],
    }, index=[f"T{t}" for t in topics_of_interest])
    print(out.to_string())


if __name__ == "__main__":
    main()
