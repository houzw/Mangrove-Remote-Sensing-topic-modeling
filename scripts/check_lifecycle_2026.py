"""Fig. 11 with and without the partially indexed 2026 in the trend statistic.

The producer runs Kendall's tau over every year present in the yearly-share matrix, which
includes 2026 (seven months of records). Every other analysis in the manuscript excludes it.
This prints the classification and the quadrant counts both ways.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from src.combined_viz import _compute_yearly_topic_proportions
from src.config import get_config

P_THRESHOLD = 0.10
config = get_config()
yearly_prop, _ = _compute_yearly_topic_proportions(config)


def summarise(frame: pd.DataFrame, label: str) -> dict:
    years = np.array([int(y) for y in frame.columns])
    rows = []
    for topic_id, row in frame.iterrows():
        vals = row.values.astype(float)
        tau, p = kendalltau(years, vals)
        rows.append({"Topic": int(topic_id), "tau": float(tau), "p": float(p),
                     "mean": float(vals.mean())})
    df = pd.DataFrame(rows)
    df["Class"] = np.where(df["p"] >= P_THRESHOLD, "Stable",
                           np.where(df["tau"] > 0, "Hot", "Cold"))
    med = df["mean"].median()
    rising = df["tau"] > 0
    above = df["mean"] > med
    out = {
        "years": f"{years.min()}-{years.max()}",
        "Hot": int((df.Class == "Hot").sum()),
        "Stable": int((df.Class == "Stable").sum()),
        "Cold": int((df.Class == "Cold").sum()),
        "rising": int(rising.sum()),
        "rising_above_median": int((rising & above).sum()),
        "falling_above_median": int((~rising & above).sum()),
        "falling_below_median": int((~rising & ~above).sum()),
        "median_mean": round(med, 4),
        "median_tau": round(float(df.tau.median()), 3),
    }
    print(f"\n{label}  (years {out['years']})")
    for k, v in out.items():
        if k != "years":
            print(f"   {k:<22} {v}")
    return out


with26 = summarise(yearly_prop, "as produced (2026 included)")
without26 = summarise(yearly_prop.drop(columns=[c for c in yearly_prop.columns if int(c) == 2026]),
                      "excluding the partially indexed 2026")
print("\ntopics whose class changes when 2026 is dropped: ", end="")
print("see tau shift below")


def classes(frame):
    years = np.array([int(y) for y in frame.columns])
    out = {}
    for tid, row in frame.iterrows():
        tau, p = kendalltau(years, row.values.astype(float))
        out[int(tid)] = "Stable" if p >= P_THRESHOLD else ("Hot" if tau > 0 else "Cold")
    return out


a = classes(yearly_prop)
b = classes(yearly_prop.drop(columns=[c for c in yearly_prop.columns if int(c) == 2026]))
changed = {t: (a[t], b[t]) for t in a if a[t] != b[t]}
print(f"{len(changed)} topics change class: {changed}")
