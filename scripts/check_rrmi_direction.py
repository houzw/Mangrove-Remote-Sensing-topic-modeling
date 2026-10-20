"""Verify every RRMI direction word and every count quoted in Sec. sec:si / sec:spatial.

Answers, from the artifacts rather than from the prose:
  P1  sign convention: which side of RRMI is under-publishing, and where Guinea-Bissau sits
      - count of negative (over-publishing) and positive (under-publishing) countries
      - the ten most negative values, and whether any of them is below the measurability threshold
  P2  how many countries are resource-rich AND below threshold, how many are measurable and
      resource-rich, and whether the two "14"s are the same set
      - the reference set of "the fourteen largest positive RRMI values"
      - how many of the 15 below-threshold area-rich countries are among them
  P3  per-country count of over-represented topics (the "between 10 and 26 of the 72" claim)

N_c comes from the full country x topic fractional matrix, recomputed in-process with the
same functions the pipeline uses (merge_country_topic + fractional_country_topic), NOT from
lq_robustness.csv, which holds only the seven reported niches.

Run: pixi run python scripts_tmp/check_rrmi_direction.py
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
    merge_country_topic,
)

PROC = Path("data/processed")
AREA_MIN = 1000.0          # km^2, the resource-rich cut used in the manuscript
NC_MIN = _N_C_MIN          # topic-assigned fractional documents (measurability)

deficit = pd.read_csv(PROC / "area_pub_deficit.csv", encoding="utf-8-sig")

# ---------------------------------------------------------------- full N_ck matrix
config = get_config()
mapping = {
    **_COUNTRY_NAME_MAPPING_DEFAULT,
    **config.country_analysis.country_name_mapping,
}
merged = merge_country_topic(config)
frac = fractional_country_topic(merged, mapping=mapping)
n_ck = frac.pivot_table(
    index="Country", columns="Topic", values="N_ck", fill_value=0.0
).reindex(sorted(frac["Topic"].unique()), axis=1)
n_ck = n_ck.rename(index={"Venzuela": "Venezuela", "Camerocn": "Cameroon",
                          "Congo.DRC": "Congo DRC"})
n_c = n_ck.sum(axis=1)
n_k = n_ck.sum(axis=0)
n_total = float(n_ck.values.sum())
lq_full = n_ck.div(n_c, axis=0).div(n_k / n_total, axis=1)

print("=" * 78)
print("Sources")
print("=" * 78)
lqr = pd.read_csv(PROC / "lq_robustness.csv", encoding="utf-8-sig")
print("topic-assigned documents with a country: %d" % len(merged))
print("countries in the fractional matrix      : %d" % len(n_c))
print("sum of N_c (fractional documents)       : %.1f" % n_total)
print("lq_robustness.csv rows / countries      : %d / %d"
      % (len(lqr), lqr["Country"].nunique()))

print()
print("=" * 78)
print("P1  RRMI sign convention (deficit = area_share - pub_share)")
print("=" * 78)
d = deficit.sort_values("deficit", ascending=False).reset_index(drop=True)
print("countries in the RRMI table : %d" % len(d))
print("positive (under-publishing) : %d" % (d["deficit"] > 0).sum())
print("negative (over-publishing)  : %d" % (d["deficit"] < 0).sum())
print()
print("top 5 positive (= most under-publishing):")
for _, r in d.head(5).iterrows():
    print("   %-28s deficit %+.4f  rank %d" % (r["country"], r["deficit"], r["rank"]))
print("ten most negative (= most over-publishing):")
for _, r in d.tail(10).iloc[::-1].iterrows():
    print("   %-28s deficit %+.4f  area %8.1f km2" % (r["country"], r["deficit"], r["area"]))

gb = d[d["country"].str.contains("Guinea", case=False, na=False)]
print()
print("Guinea / Guinea-Bissau rows (sign decides which side of the ledger they sit on):")
print(gb[["country", "area", "pubs", "deficit", "rank"]].to_string(index=False))

top14 = d.head(14)
print()
print("top-14 positive RRMI hold %.2f%% of area and %.2f%% of pubs"
      % (100 * top14["area_share"].sum(), 100 * top14["pub_share"].sum()))

print()
print("=" * 78)
print("P1/P3  Measurability split (N_c from the full matrix)")
print("=" * 78)
area = deficit.set_index("country")["area"]
deficit_by_country = deficit.set_index("country")["deficit"]

measurable = n_c[n_c >= NC_MIN]
below = n_c[n_c < NC_MIN]
print("measurable (N_c >= %d): %d ; below threshold: %d" % (NC_MIN, len(measurable), len(below)))

area_of = {c: float(area.get(c, 0.0)) for c in n_c.index}
rich_below = [c for c in below.index if area_of[c] >= AREA_MIN]
rich_meas = [c for c in measurable.index if area_of[c] >= AREA_MIN]
print("resource-rich (>= %.0f km2) AND below threshold: %d" % (AREA_MIN, len(rich_below)))
print("   " + ", ".join(sorted(rich_below)))
print("resource-rich AND measurable: %d" % len(rich_meas))
print("   " + ", ".join(sorted(rich_meas)))

area_share_map = deficit.set_index("country")["area_share"]
pub_share_map = deficit.set_index("country")["pub_share"]
rich_below_df = pd.DataFrame(
    {"N_c": below.reindex(rich_below).round(2),
     "area": [area_of[c] for c in rich_below],
     "area_share": [float(area_share_map.get(c, 0)) for c in rich_below],
     "pub_share": [float(pub_share_map.get(c, 0)) for c in rich_below],
     "deficit": [float(deficit_by_country.get(c, 0)) for c in rich_below]},
    index=rich_below,
).sort_values("area", ascending=False)
print()
print(rich_below_df.to_string())
print("   sum area share = %.4f (%.2f%%) ; sum pub share = %.4f (%.2f%%)"
      % (rich_below_df["area_share"].sum(), 100 * rich_below_df["area_share"].sum(),
         rich_below_df["pub_share"].sum(), 100 * rich_below_df["pub_share"].sum()))

print()
print("=" * 78)
print("P2  The two '14's")
print("=" * 78)
top14_names = set(top14["country"])
print("set A = top-14 by RRMI (of %d):" % len(d))
print("   " + ", ".join(sorted(top14_names)))
print("set B = the %d measurable resource-rich countries:" % len(rich_meas))
print("   " + ", ".join(sorted(rich_meas)))
print("identical sets: %s ; overlap %d of 14"
      % (set(top14_names) == set(rich_meas), len(top14_names & set(rich_meas))))

print()
print("=" * 78)
print("P2  Claims to check")
print("=" * 78)
rich_below_in_top14 = [c for c in rich_below if c in top14_names]
print("below-threshold area-rich countries among the top-14 positive RRMI: %d"
      % len(rich_below_in_top14))
print("   " + ", ".join(sorted(rich_below_in_top14)))

below_positive = [c for c in rich_below if deficit_by_country.get(c, 0) > 0]
print("below-threshold area-rich that are UNDER-publishing (deficit>0): %d of %d"
      % (len(below_positive), len(rich_below)))
below_negative = [c for c in rich_below if deficit_by_country.get(c, 0) < 0]
print("   of which over-publishing (deficit<0): %s" % (below_negative or "none"))

ten_most_negative = list(d.tail(10)["country"])
ten_below = [c for c in ten_most_negative if c in set(below.index)]
print("ten most negative RRMI values that are BELOW the threshold: %d %s"
      % (len(ten_below), ten_below or "(none)"))
neg = set(d.loc[d["deficit"] < 0, "country"])
print("below-threshold countries that are negative-RRMI: %d of %d"
      % (len(set(below.index) & neg), len(below)))

print()
print("=" * 78)
print("P3  Per-country over-represented topics (LQ > 1) among the 27 measurable")
print("=" * 78)
n_over = (lq_full.loc[measurable.index] > 1.0).sum(axis=1).astype(int)
print("n = %d ; range %d to %d ; median %.0f"
      % (len(n_over), n_over.min(), n_over.max(), n_over.median()))
print("countries at the minimum (%d topics): %s"
      % (n_over.min(), ", ".join(sorted(n_over[n_over == n_over.min()].index))))
print("countries at the maximum (%d topics): %s"
      % (n_over.max(), ", ".join(sorted(n_over[n_over == n_over.max()].index))))
print()
print("full distribution:")
print(n_over.sort_values(ascending=False).to_string())
print()
print("the seven reported niches, over-represented topic count:")
for c, t in _NICHES:
    print("   %-14s N_c %6.2f  %2d topics at LQ>1   reported T%d (LQ %.2f, N_ck %.2f)"
          % (c, n_c[c], n_over[c], t, lq_full.loc[c, t], n_ck.loc[c, t]))
print()
print("cross-check vs lq_robustness.csv observed_LQ:")
lq_cols = [int(x) for x in lq_full.columns]
for _, r in lqr.iterrows():
    c, t = r["Country"], int(r["Topic"])
    if c in lq_full.index and t in lq_cols:
        print("   %-14s T%-3d csv %.3f   recomputed %.3f" % (c, t, r["observed_LQ"], lq_full.loc[c, t]))

print()
print("=" * 78)
print("P2  The seven reported niches: top-7 positive RRMI among the measurable resource-rich")
print("=" * 78)
ranked = sorted(rich_meas, key=lambda c: -deficit_by_country.get(c, 0))
for i, c in enumerate(ranked):
    star = "  <== reported" if i < 7 else ""
    print("   %2d. %-14s RRMI %+.4f  N_c %6.2f%s"
          % (i + 1, c, deficit_by_country.get(c, 0), n_c[c], star))
print("   frozen _NICHES country set: %s" % sorted({c for c, _ in _NICHES}))
print("   matches top-7: %s" % (set(ranked[:7]) == {c for c, _ in _NICHES}))


print()
print("=" * 78)
print("P3  Cross-denominator check: does the 17 sit inside the 76?")
print("=" * 78)
deficit_countries = set(deficit["country"])
below_set = set(below.index)
print("matrix countries %d ; RRMI table %d ; below-threshold %d"
      % (len(n_c), len(deficit), len(below_set)))
print("below-threshold that have an area row : %d"
      % len(below_set & deficit_countries))
print("below-threshold with NO area row      : %d -> %s"
      % (len(below_set - deficit_countries),
         ", ".join(sorted(below_set - deficit_countries))))
neg_rrmi = set(deficit.loc[deficit["deficit"] < 0, "country"])
seventeen = sorted(below_set & neg_rrmi)
print("below-threshold AND negative-RRMI     : %d" % len(seventeen))
print("   all inside the 76: %s" % set(seventeen).issubset(deficit_countries))
areas = [float(area[c]) for c in seventeen]
print("   max area %.1f km2 ; n with area >= 1000: %d"
      % (max(areas), sum(a >= 1000 for a in areas)))
print("   largest: %s" % ", ".join(
    "%s %.1f" % (c, area[c]) for c in sorted(seventeen, key=lambda c: -area[c])[:5]))

print()
print("=" * 78)
print("P3b The 76/110 ledger")
print("=" * 78)
matrix_set = set(n_c.index)
meas_set = set(measurable.index)
meas_with_area = meas_set & deficit_countries
meas_without_area = meas_set - deficit_countries
rr_only = deficit_countries - matrix_set          # in the RRMI table, absent from the matrix
print("matrix countries (topic-assigned)      : %d" % len(matrix_set))
print("RRMI table countries (area + pubs)     : %d" % len(deficit_countries))
print("  below-threshold (83) WITH area       : %d" % len(below_set & deficit_countries))
print("  measurable (27) WITH area            : %d" % len(meas_with_area))
print("     measurable WITHOUT area           : %d -> %s"
      % (len(meas_without_area), ", ".join(sorted(meas_without_area)) or "(none)"))
print("  in the RRMI table but NOT in the matrix: %d" % len(rr_only))
print("     (mangrove area and publications, but no topic-assigned document)")
for c in sorted(rr_only, key=lambda c: -float(deficit.set_index("country")["pubs"][c])):
    row = deficit.set_index("country").loc[c]
    print("       %-22s area %8.1f km2   pubs %6.2f frac   RRMI %+.5f   N_c = 0"
          % (c, row["area"], row["pubs"], row["deficit"]))
print()
lhs = (len(below_set & deficit_countries) + len(meas_with_area) + len(rr_only))
print("ledger: %d (below-thr. with area) + %d (measurable with area) + %d (RRMI-only) = %d"
      % (len(below_set & deficit_countries), len(meas_with_area), len(rr_only), lhs))
print("balances against the RRMI table: %s" % (lhs == len(deficit_countries)))
print("also: 110 matrix countries = %d measurable + %d below-threshold = %d"
      % (len(meas_set), len(below_set), len(meas_set) + len(below_set)))
