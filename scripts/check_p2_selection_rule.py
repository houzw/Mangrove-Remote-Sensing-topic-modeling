"""Third pass: what selection rule actually yields the seven tab:lq niches, and
reproduce the remaining Sec. spatial correlation numbers."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_config  # noqa: E402
from src.lq_robustness import _NICHES  # noqa: E402
from src.spatial_topic_analysis import (  # noqa: E402
    _COUNTRY_NAME_MAPPING_DEFAULT,
    fractional_country_topic,
    merge_country_topic,
)

UNFOLD = {"Venzuela": "Venezuela", "Camerocn": "Cameroon",
          "Congo.DRC": "Congo DRC", "Solomon Is.": "Solomon Islands"}


def main() -> None:
    config = get_config()
    paths = config.paths
    mapping = {**_COUNTRY_NAME_MAPPING_DEFAULT,
               **config.country_analysis.country_name_mapping}

    merged = merge_country_topic(config)
    frac = fractional_country_topic(merged, mapping=mapping)
    n_ck = frac.pivot_table(index="Country", columns="Topic", values="N_ck",
                            fill_value=0.0).reindex(sorted(frac["Topic"].unique()), axis=1)
    n_c = n_ck.sum(axis=1)
    n_k = n_ck.sum(axis=0)
    N = float(n_ck.values.sum())
    lq = n_ck.div(n_c, axis=0).div(n_k / N, axis=1)

    print("=" * 78)
    print("A. For each reported niche topic: is the country its largest N_ck holder?")
    print("=" * 78)
    rows = []
    for c, t in _NICHES:
        order = n_ck[t].sort_values(ascending=False)
        rank_c = list(order.index).index(c) + 1
        rows.append({
            "country": c, "topic": f"T{t}",
            "N_ck": round(float(n_ck.loc[c, t]), 2),
            "country_N_ck_rank_in_topic": rank_c,
            "top1": order.index[0], "top2": order.index[1], "top3": order.index[2],
            "country_LQ": round(float(lq.loc[c, t]), 2),
            "country_LQ_rank_in_topic": int(
                (lq[t] > lq.loc[c, t]).sum()) + 1,
        })
    print(pd.DataFrame(rows).to_string(index=False))

    print()
    print("=" * 78)
    print("B. Measurable (N_c>=10) AND area>=1000: who, and their max-LQ topic")
    print("=" * 78)
    deficit = pd.read_csv(paths.processed_dir / "area_pub_deficit.csv")
    nc = n_c.rename(index=UNFOLD)
    deficit["N_c"] = deficit["country"].map(nc)
    m = deficit[(deficit["N_c"] >= 10) & (deficit["area"] >= 1000)].copy()
    print(f"n = {len(m)}")
    lq_named = lq.rename(index=UNFOLD)
    m["maxLQ_topic"] = [int(lq_named.loc[r.country].idxmax()) for r in m.itertuples()]
    m["maxLQ"] = [round(float(lq_named.loc[r.country].max()), 2) for r in m.itertuples()]
    m["n_topics_LQ>1"] = [(lq_named.loc[r.country] > 1).sum() for r in m.itertuples()]
    m["is_top_Nck_holder"] = [
        n_ck[int(lq_named.loc[r.country].idxmax())].idxmax()
        .replace("Venzuela", "Venezuela").replace("Camerocn", "Cameroon") == r.country
        for r in m.itertuples()
    ]
    print(m[["country", "area", "N_c", "maxLQ_topic", "maxLQ", "n_topics_LQ>1",
             "is_top_Nck_holder"]].sort_values("maxLQ", ascending=False).to_string(index=False))

    print()
    print("=" * 78)
    print("C. Sec. spatial correlation claims (from multi_indicator_consistency.csv)")
    print("=" * 78)
    mi = pd.read_csv(paths.processed_dir / "multi_indicator_consistency.csv")
    print(mi.columns.tolist())
    rho = mi["deficit"].corr(mi["pub_density"], method="spearman")
    print(f"Spearman(RRMI, pub_density) = {rho:.4f}  n={len(mi)}")
    rho2 = mi["deficit"].corr(mi["citation_gap"], method="spearman")
    print(f"Spearman(RRMI, citation_gap) = {rho2:.4f}  n={len(mi)}")
    top = mi.sort_values("pub_density", ascending=False).head(3)
    print(top[["Country", "area", "pubs", "pub_density"]].to_string(index=False))

    print()
    print("=" * 78)
    print("D. Nigeria x T50 document")
    print("=" * 78)
    from src.utils import canonical_countries
    sub = merged.copy()
    hit = []
    for i, r in sub.iterrows():
        countries = canonical_countries(str(r["CU"]), fold=mapping,
                                        drop=__import__("src.spatial_topic_analysis",
                                                        fromlist=["_CU_FALSE_POSITIVES"])._CU_FALSE_POSITIVES)
        if "Nigeria" in countries and int(r["Topic"]) == 50:
            hit.append(r)
    print(f"documents with Nigeria and topic 50: {len(hit)}")
    cleaned = pd.read_parquet(paths.processed_dir / "cleaned_data.parquet")
    for r in hit:
        di = int(r["doc_index"])
        row = cleaned.iloc[di]
        print("  doc_index:", di)
        print("  TI:", str(row.get("TI"))[:200])
        print("  PY:", row.get("PY"))
        print("  CU:", str(row.get("CU"))[:200])


if __name__ == "__main__":
    main()
