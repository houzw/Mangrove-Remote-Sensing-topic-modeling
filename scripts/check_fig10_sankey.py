"""Audit the Fig. 10 (topic Sankey) claims: periods, weighting, and the first window.

Checks:
  * how many period columns the Sankey actually has (caption says "five phases")
  * per window: corpus documents, topic-assigned documents, distinct topics
  * the 1972-1984 column's single-node (T37 = 1.0) provenance
  * what the alternative phase-aligned binning would give
"""

from __future__ import annotations

import pandas as pd

PROC = "data/processed/"
props = pd.read_csv(PROC + "topic_evolution_proportions.csv", index_col=0)
print("Sankey period columns:", list(props.columns))
print("topic rows kept     :", list(props.index))
first = props.columns[0]
top = props[first].sort_values(ascending=False)
print(f"first window {first}: top topic T{top.index[0]} = {top.iloc[0]:.3f}, "
      f"non-zero topics = {(props[first] > 0).sum()}")

docs = pd.read_csv(PROC + "cleaned_data.csv", encoding="utf-8-sig", usecols=["PY"])
dt = pd.read_csv(PROC + "doc_topic_dist.csv", encoding="utf-8-sig")
print("\ndoc_topic_dist columns:", list(dt.columns)[:6], " rows:", len(dt))
year = docs["PY"] if len(docs) == len(dt) else None
if year is None:
    raise SystemExit("cleaned_data and doc_topic_dist differ in length; join by index needed")

df = pd.DataFrame({"PY": year.values, "Topic": dt["Topic"].values})
assigned = df[df["Topic"] != -1]

windows = {
    "auto-binned (as plotted)": [(1972, 1984), (1985, 1998), (1999, 2012), (2013, 2026)],
    "manuscript phases": [(1972, 1984), (1985, 1998), (1999, 2012), (2013, 2022), (2023, 2025)],
    "phases incl. 2026": [(1972, 1984), (1985, 1998), (1999, 2012), (2013, 2022), (2023, 2026)],
}
for name, wins in windows.items():
    print(f"\n{name}")
    print("   window        corpus  assigned  topics  top topic (share of assigned)")
    for lo, hi in wins:
        allw = df[(df["PY"] >= lo) & (df["PY"] <= hi)]
        asw = assigned[(assigned["PY"] >= lo) & (assigned["PY"] <= hi)]
        if len(asw):
            vc = asw["Topic"].value_counts()
            print(f"   {lo}-{hi}   {len(allw):6d}  {len(asw):8d}  {asw['Topic'].nunique():6d}  "
                  f"T{vc.index[0]} ({vc.iloc[0] / len(asw):.2f})")
        else:
            print(f"   {lo}-{hi}   {len(allw):6d}  {len(asw):8d}  {0:6d}  --")

print("\nper-year assigned documents, 1972-1990:")
print(assigned[assigned["PY"] <= 1990].groupby("PY").size().to_string())
