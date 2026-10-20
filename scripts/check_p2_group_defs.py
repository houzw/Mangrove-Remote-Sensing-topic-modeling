"""Second verification pass: reproduce the numeric claims around Sec. si and
the discussion's "fourteen resource-rich countries" sentence, and pin down the
group definitions.

All numbers are read from data/processed/area_pub_deficit.csv and the same
topic-assigned fractional matrix used by src/lq_robustness.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_config  # noqa: E402
from src.spatial_topic_analysis import (  # noqa: E402
    _COUNTRY_NAME_MAPPING_DEFAULT,
    fractional_country_topic,
    merge_country_topic,
)

# Reverse of the config fold: canonical (folded) name -> HGMF/deficit-table name
UNFOLD = {
    "Venzuela": "Venezuela",
    "Camerocn": "Cameroon",
    "Congo.DRC": "Congo DRC",
    "Solomon Is.": "Solomon Islands",
}


def main() -> None:
    config = get_config()
    paths = config.paths

    deficit = pd.read_csv(paths.processed_dir / "area_pub_deficit.csv")
    n_countries = len(deficit)
    tot_area_share = deficit["area_share"].sum()
    tot_pub_share = deficit["pub_share"].sum()
    print(f"deficit table: {n_countries} countries; "
          f"sum area_share={tot_area_share:.4f} sum pub_share={tot_pub_share:.4f}")
    print(f"HGMF global area in table = {deficit['area'].sum():,.1f} km2")

    top14 = deficit.sort_values("deficit", ascending=False).head(14)
    print("\n--- top-14 by RRMI (manuscript: 63.5% area / 16.7% pubs) ---")
    print(f"area share = {top14['area_share'].sum():.4f} "
          f"({100 * top14['area_share'].sum():.1f}%)")
    print(f"pub  share = {top14['pub_share'].sum():.4f} "
          f"({100 * top14['pub_share'].sum():.1f}%)")
    print(top14[["country", "area", "pubs", "area_share", "pub_share",
                 "deficit"]].to_string(index=False))

    print("\n--- 'resource-rich countries below the N_c<10 threshold' ---")
    for label, mask in [
        ("deficit-table pubs < 10 AND area >= 1000",
         (deficit["pubs"] < 10) & (deficit["area"] >= 1000)),
        ("deficit-table pubs < 10 AND area >= 940 (incl. Nicaragua)",
         (deficit["pubs"] < 10) & (deficit["area"] >= 940)),
        ("deficit-table pubs < 10 (any area)",
         deficit["pubs"] < 10),
    ]:
        g = deficit[mask]
        print(f"  [{label}] n={len(g)}  area={100 * g['area_share'].sum():.1f}%  "
              f"pubs={100 * g['pub_share'].sum():.2f}%")
        if len(g) <= 20:
            print("     ", ", ".join(g.sort_values("area", ascending=False)["country"]))

    # --- same group keyed on the topic-assigned fractional N_c used by the LQ ---
    mapping = {
        **_COUNTRY_NAME_MAPPING_DEFAULT,
        **config.country_analysis.country_name_mapping,
    }
    merged = merge_country_topic(config)
    frac = fractional_country_topic(merged, mapping=mapping)
    n_ck = frac.pivot_table(index="Country", columns="Topic", values="N_ck",
                            fill_value=0.0)
    n_c = n_ck.sum(axis=1)
    nc = n_c.rename(index=UNFOLD)
    deficit["N_c_topic"] = deficit["country"].map(nc)
    print(f"\ntopic-assigned N_c matched onto the deficit table: "
          f"{deficit['N_c_topic'].notna().sum()}/{len(deficit)} countries")
    unm = deficit.loc[deficit["N_c_topic"].isna(), "country"].tolist()
    print(f"  unmatched: {unm}")

    for label, mask in [
        ("topic-assigned N_c < 10 AND area >= 1000",
         (deficit["N_c_topic"] < 10) & (deficit["area"] >= 1000)),
        ("full-corpus pubs < 10 AND area >= 1000",
         (deficit["pubs"] < 10) & (deficit["area"] >= 1000)),
    ]:
        g = deficit[mask]
        print(f"  [{label}] n={len(g)}  area={100 * g['area_share'].sum():.1f}%  "
              f"pubs={100 * g['pub_share'].sum():.2f}%")
        print("     ", ", ".join(g.sort_values("area", ascending=False)["country"]))

    print("\n--- most-negative-RRMI group (manuscript: 'the ten countries') ---")
    ten = deficit.sort_values("deficit").head(10)
    print(ten[["country", "area", "pubs", "deficit"]].to_string(index=False))
    print(f"  of these, area < 350 km2: "
          f"{sorted(ten.loc[ten['area'] < 350, 'country'])}")
    print(f"  of these, area > 1000 km2: "
          f"{sorted(ten.loc[ten['area'] > 1000, 'country'])}")

    print("\n--- correlations claimed in Sec. spatial ---")
    d = deficit.dropna(subset=["pub_density"])
    if len(d):
        rho = d["deficit"].corr(d["pub_density"], method="spearman")
        print(f"  Spearman(RRMI, pubs per 1e3 km2) = {rho:.3f}  (n={len(d)})")
    c = deficit.dropna(subset=["citation_gap"])
    if len(c):
        rho2 = c["deficit"].corr(c["citation_gap"], method="spearman")
        print(f"  Spearman(RRMI, citation gap)      = {rho2:.3f}  (n={len(c)})")
    if "pub_density" in deficit.columns:
        s = deficit.sort_values("pub_density", ascending=False).head(3)
        print("  top pub_density:")
        print(s[["country", "area", "pubs", "pub_density"]].to_string(index=False))


if __name__ == "__main__":
    main()
