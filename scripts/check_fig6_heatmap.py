"""Verify the Figure 6 heatmap against its caption (2026-09-24 revision).

Checks the artifact produced by ``src.combined_viz.plot_topic_year_heatmap``:
  1. row set = 35 by cumulative share (complete years) + the latest-year leaders
     the cumulative cut omitted, and that those extras are the starred rows;
  2. the column axis covers every calendar year to the last complete year, with
     the blank years reported;
  3. the colour ceiling actually used and how many cells sit above it;
  4. the figure is designed at printed width (6.50 in = \\columnwidth) so it is
     placed at scale ~1.0, and no ink touches a canvas edge (no clipped label).

Run: pixi run python scripts_tmp/check_fig6_heatmap.py
"""

from pathlib import Path

import numpy as np
from PIL import Image

from src.combined_viz import (
    LAST_COMPLETE_YEAR,
    _PRINT_WIDTH_IN,
    _compute_yearly_topic_proportions,
    _load_topic_labels,
    _prepare_topic_year_matrix,
)
from src.config import get_config

config = get_config()
figure = config.paths.figures_dir / "topic_year_heatmap.png"

proportions, assigned_per_year = _compute_yearly_topic_proportions(config)
data_years = sorted(int(y) for y in proportions.columns if int(y) <= LAST_COMPLETE_YEAR)
frame = proportions.reindex(columns=data_years, fill_value=0.0)
frame = frame[(frame > 0).sum(axis=1) >= 3]

totals = frame.sum(axis=1).sort_values(ascending=False)
selected = list(totals.index[:35])
latest = data_years[-1]
leaders = list(frame[latest].sort_values(ascending=False).head(5).index)
added = [t for t in leaders if t not in selected]

print("-- 1. rows -------------------------------------------------")
print("drawn rows      : %d (35 cumulative + %d latest-year leaders)" % (len(selected) + len(added), len(added)))
print("starred rows    : %s" % sorted(int(t) for t in added))
print("2025 leaders    : %s  (all present: %s)" % ([int(t) for t in leaders], set(leaders) <= set(selected) | set(added)))
labels = _load_topic_labels(config)
for t in added:
    print("   *T%-3d %s" % (int(t), labels.get(int(t), "")))

print("\n-- 2. year axis --------------------------------------------")
print("data-year columns: %d  (%d-%d)" % (len(data_years), data_years[0], data_years[-1]))
print("2026 drawn       : %s" % (2026 in data_years))

print("\n-- 3. colour scale -----------------------------------------")
data = frame.loc[selected + added].values
positive = data[data > 0]
vmax = float(min(data.max(), np.percentile(positive, 95)))
print("data max        : %.3f" % data.max())
print("colour ceiling  : %.3f (95th pct of non-zero cells)" % vmax)
print("cells above it  : %d  (all drawn at the ceiling)" % int((data > vmax).sum()))

print("\n-- 4. figure geometry --------------------------------------")
figure = config.paths.figures_dir / "topic_year_heatmap.png"
with Image.open(figure) as im:
    width_px, height_px = im.size
    ink = np.array(im.convert("L")) < 200
gaps = (
    int(np.where(ink.any(axis=0))[0].min()),
    int(width_px - 1 - np.where(ink.any(axis=0))[0].max()),
    int(np.where(ink.any(axis=1))[0].min()),
    int(height_px - 1 - np.where(ink.any(axis=1))[0].max()),
)
print("file          : %s" % figure.name)
print("size          : %d x %d px = %.2f x %.2f in at 300 dpi" % (width_px, height_px, width_px / 300, height_px / 300))
print("placement    : %.3f x print width -> 6.5 pt labels print at %.1f pt"
      % (_PRINT_WIDTH_IN / (width_px / 300), 6.5 * _PRINT_WIDTH_IN / (width_px / 300)))
print("edge gaps px  : left=%d right=%d top=%d bottom=%d (all must be > 0)" % gaps)

prep = _prepare_topic_year_matrix(config)
print("\n-- 5. no break bands (author decision 2026-09-24) ----------")
print("break-slot machinery present: %s" % ("gap_positions" in prep))
print("omitted spans (caption only): %s" % prep["missing_spans"])
print("columns drawn               : %d (every data year)" % len(prep["years"]))

print("\n-- 6. caption claims vs artifact ---------------------------")
# The caption is the artifact's description, so read it from the manuscript and
# verify it still uses the agreed wording (2026-09-24: "topic-assigned documents",
# not the bare "assigned documents" of the BERTopic API).
tex = Path("manuscript/els-cas-templates/manuscript-v3.tex").read_text(encoding="utf-8")
_tex_before_label = tex.split(r"\label{fig:topic_heatmap}")[0]
caption = _tex_before_label.rpartition(r"\caption{")[2]
print("caption length : %d chars" % len(caption))
print("uses 'topic-assigned': %s" % ("topic-assigned" in caption))

checks = {
    "caption says 35 cumulative rows": len(selected) == 35,
    "caption says four 2025 leaders added": len(added) == 4,
    "caption names T4, T8, T16, T33": sorted(int(t) for t in added) == [4, 8, 16, 33],
    "caption says 39 data-year columns": len(data_years) == 39,
    "caption names 1973-1984, 1987-1988, 1990": prep["missing_spans"] == ["1973\u20131984", "1987\u20131988", "1990"],
    "caption no longer promises grey bands": "gap_positions" not in prep,
    "caption says 2026 excluded": 2026 not in data_years,
    "caption says ceiling 0.24": round(vmax, 2) == 0.24,
    "caption uses 'topic-assigned documents'": "topic-assigned documents" in caption,
    "caption gives the reason for the omitted years": "contain no topic-assigned document" in caption,
    "caption defines the cumulative-share criterion": "sum of a topic's yearly shares" in caption,
    "caption discloses that only 39 of 72 topics are drawn": "remaining 33 topics are not shown" in caption,
    "caption distinguishes the row clustering and gives its metric": (
        "Ward linkage on the Euclidean distances between their yearly-share profiles" in caption
    ),
    "caption explains the every-third-column year ticks": "every third column" in caption,
    "figure placed at scale ~1.0": abs(_PRINT_WIDTH_IN - width_px / 300) < 0.01,
    "no clipped label (all four edges)": min(gaps) > 0,
}
for name, ok in checks.items():
    print("   [%s] %s" % ("ok" if ok else "FAIL", name))
print("\nALL CHECKS PASS" if all(checks.values()) else "\nSOME CHECKS FAILED")
