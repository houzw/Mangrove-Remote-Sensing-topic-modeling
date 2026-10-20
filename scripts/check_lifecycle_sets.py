"""Are the two under-publishing sets the same? (top-14 by RRMI vs the 15 below-threshold)

§4.3 quotes the 15 area-rich measurability-limited countries at 26.8% of global mangrove area;
§4.5 quotes the top-14 by RRMI at 63.5%. This prints each set, its area share and their overlap.
"""

from __future__ import annotations

import pandas as pd

d = pd.read_csv("data/processed/area_pub_deficit.csv", encoding="utf-8-sig")
d = d.sort_values("deficit", ascending=False).reset_index(drop=True)

top14 = d.head(14)
print("top-14 by positive RRMI (deficit):")
for _, r in top14.iterrows():
    print(f"   {r['country']:<22} area_share={r['area_share']:.4f}  deficit={r['deficit']:+.4f}")
print(f"   sum area_share = {top14['area_share'].sum():.4f}  ({top14['area_share'].sum():.1%})")
print(f"   15th = {d.iloc[14]['country']} at {d.iloc[14]['deficit']:+.4f}")

# The 15 area-rich below-threshold countries (re-derived in the §si round by
# scripts_tmp/check_rrmi_direction.py: N_c < 10 in the 110-country topic matrix and >= 1,000 km^2)
below15 = ["Papua New Guinea", "Myanmar", "Cuba", "Venezuela", "Mozambique", "Guinea-Bissau",
           "Madagascar", "Guinea", "Gabon", "Panama", "Senegal", "Ecuador", "Cameroon",
           "Sierra Leone", "Tanzania"]
sub = d[d["country"].isin(below15)]
print(f"\nbelow-threshold area-rich set: {len(sub)} of the 15 found in the RRMI table")
print(f"   sum area_share = {sub['area_share'].sum():.4f}  ({sub['area_share'].sum():.1%})")
missing = sorted(set(below15) - set(sub["country"]))
print(f"   not in the RRMI table: {missing}")

inter = sorted(set(top14["country"]) & set(below15))
print(f"\noverlap with the top-14: {len(inter)} -> {inter}")
print("top-14 members NOT in the below-threshold set:",
      sorted(set(top14['country']) - set(below15)))
print(f"\nglobal area denominator: {d['area'].sum():,.0f} km2")
