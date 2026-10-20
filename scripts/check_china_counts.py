"""Adjudicate §4.3's China numbers: full-count vs fractional vs topic-assigned N_c.

Reported in the text: "China published 570 publications (24% of the corpus)".
Reported in Fig. 7's row label: "China (325)".
Both should be right, on different counting bases; this prints each basis for China and
Indonesia (the comparison in the same sentence) so the text can name its basis.
"""

from __future__ import annotations

import pandas as pd

from src.config import get_config
from src.country_topic_heatmap import build_country_topic_matrix
from src.utils import canonical_countries

config = get_config()
proc = config.paths.processed_dir

clean = pd.read_csv(proc / config.preprocessing.output_file, encoding="utf-8-sig")
dt = pd.read_csv(proc / config.topic_modeling.output.doc_topic_dist, encoding="utf-8-sig")

print("cleaned_data columns:", [c for c in clean.columns][:8], "...")

# --- full count: each publication credits a country once --------------------
country_col = next(c for c in ("countries", "Countries", "cu_countries", "CU") if c in clean.columns)
assigned = (dt["Topic"] != -1).values
full = {}
frac_all = {}
frac_assigned = {}
for name, cols in (("full-count", None), ("fractional (full corpus)", None), ("fractional (topic-assigned)", None)):
    pass

for i, row in clean.iterrows():
    cs = canonical_countries(row.get(country_col))
    if not cs:
        continue
    w = 1.0 / len(cs)
    for c in cs:
        full[c] = full.get(c, 0) + 1
        frac_all[c] = frac_all.get(c, 0.0) + w
        if assigned[i]:
            frac_assigned[c] = frac_assigned.get(c, 0.0) + w

n = len(clean)
print(f"\ncorpus = {n} publications; topic-assigned = {int(assigned.sum())} "
      f"({assigned.mean():.1%}); unassigned = {int((~assigned).sum())}")

print(f"\n{'country':<10}{'full-count':>12}{'share of corpus':>17}"
      f"{'frac (full)':>13}{'frac (assigned) = N_c':>24}")
for c in ("China", "Indonesia", "United States", "India"):
    if c in full:
        print(f"{c:<10}{full[c]:>12}{full[c]/n:>16.1%}{frac_all.get(c, 0):>13.1f}"
              f"{frac_assigned.get(c, 0):>24.1f}")

# --- what Fig. 7 actually labels -------------------------------------------
block = build_country_topic_matrix(config)
print("\nFig. 7 row label for China:", [lab for lab in block["row_labels"] if "China" in lab])
print("Fig. 7 row label for Indonesia:",
      [lab for lab in block["row_labels"] if "Indonesia" in lab])
print("40.3% zero-cell / 38.0% coverage claims: matrix", block["matrix"].shape)
