"""Fourth pass: for the seven tab:lq countries, list the top topics by LQ so we
can see whether the reported topic is the country's maximum-LQ topic (i.e.
whether a mechanical rule reproduces the printed table)."""

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
    load_topic_labels,
    merge_country_topic,
)

UNFOLD = {"Venzuela": "Venezuela", "Camerocn": "Cameroon",
          "Congo.DRC": "Congo DRC"}


def main() -> None:
    config = get_config()
    paths = config.paths
    mapping = {**_COUNTRY_NAME_MAPPING_DEFAULT,
               **config.country_analysis.country_name_mapping}
    merged = merge_country_topic(config)
    frac = fractional_country_topic(merged, mapping=mapping)
    n_ck = frac.pivot_table(index="Country", columns="Topic", values="N_ck",
                            fill_value=0.0).reindex(sorted(frac["Topic"].unique()), axis=1)
    n_ck = n_ck.rename(index=UNFOLD)
    n_c = n_ck.sum(axis=1)
    n_k = n_ck.sum(axis=0)
    N = float(n_ck.values.sum())
    lq = n_ck.div(n_c, axis=0).div(n_k / N, axis=1)
    labels = load_topic_labels(config)
    clusters = pd.read_csv(paths.processed_dir / "topic_clusters.csv")
    cl = clusters.set_index("Topic")["Cluster"].to_dict()

    deficit = pd.read_csv(paths.processed_dir / "area_pub_deficit.csv")
    nc_map = n_c.to_dict()
    print("RRMI-ranked measurable & area>=1000 group (selection rule check):")
    d = deficit.copy()
    d["N_c"] = d["country"].map(nc_map)
    g = d[(d["N_c"] >= 10) & (d["area"] >= 1000)].sort_values(
        "deficit", ascending=False)
    print(g[["country", "area", "deficit", "N_c"]].head(10).to_string(index=False))
    print()

    niches = dict(_NICHES)
    for c in sorted({c for c, _ in _NICHES}):
        cn = {"Venezuela": "Venzuela"}.get(c, c)
        row = lq.loc[c].sort_values(ascending=False).head(6)
        print(f"--- {c}  (N_c={n_c[c]:.2f})  reported topic = T{niches[c]} ---")
        for t, v in row.items():
            mark = "  <== reported" if int(t) == niches[c] else ""
            print(f"    T{int(t):>2} LQ={v:6.2f}  N_ck={n_ck.loc[c, t]:5.2f}  "
                  f"cluster {cl.get(int(t), '?')}  {labels.get(int(t), '?')}{mark}")
        print()


if __name__ == "__main__":
    main()
