"""Reconcile the text percentages with the Fig. 9 (stacked cluster) series.

The figure plots a centred 5-year moving average (truncated at the edges, then
renormalised); the manuscript quotes raw annual shares. This prints both, so the
caption can state the basis explicitly.

Run: pixi run python -m scripts_tmp.check_fig9_shares
"""

from __future__ import annotations

from src.cluster_evolution_stacked import PHASES, cluster_yearly_shares
from src.config import get_config
from src.topic_clusters import BASE_YEAR, CLUSTERS

config = get_config()
letters = [c["letter"] for c in CLUSTERS]

for window in (0, 5):
    years, mat, _, _ = cluster_yearly_shares(config, window=window)
    label = "raw annual shares" if window == 0 else "centred 5-yr MA (edge-truncated)"
    print(f"\n{label}")
    head = "  year " + " ".join(f"{x:>7s}" for x in letters) + "     sum"
    print(head)
    for y in (2010, 2015, 2020, 2023, 2024, 2025):
        j = list(years).index(y)
        row = " ".join(f"{mat[i, j] * 100:6.1f}%" for i in range(len(letters)))
        print(f"  {y} {row}  {mat[:, j].sum():.4f}")

# how many raw years enter each smoothed point (edge truncation)
years, _, _, _ = cluster_yearly_shares(config, window=0)
print("\nwindow width actually used at the ends:",
      f"1972-1973 -> 3 raw years, 1974-2022 -> 5, 2023-{BASE_YEAR} -> 3")
print("phase windows in the producer:", PHASES, " (no (v) boundary drawn)")
