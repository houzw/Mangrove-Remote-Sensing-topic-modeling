"""Check which "largest cumulative share" definition the figure actually uses.

Fig. 6 selects rows with
    totals = yearly_prop.sum(axis=1)
where yearly_prop[t, y] = (topic-assigned documents of t in y) / (topic-assigned
documents in y). That is the SUM OF YEARLY SHARES: every year carries the same
weight regardless of how large it was. The alternative an author might assume is
the cumulative topic-assigned document COUNT, which lets big years dominate.

This script prints both rankings and how much the top-35 differ.
"""

from pathlib import Path

from src.combined_viz import (
    LAST_COMPLETE_YEAR,
    _compute_yearly_topic_proportions,
    _load_topic_labels,
)
from src.config import get_config

config = get_config()
proc: Path = config.paths.processed_dir

proportions, assigned_per_year = _compute_yearly_topic_proportions(config)
years = sorted(int(y) for y in proportions.columns if int(y) <= LAST_COMPLETE_YEAR)
prop = proportions.reindex(columns=years, fill_value=0.0)

# Rebuild the counts behind the proportions: count = share * assigned(y).
counts = prop.mul(assigned_per_year.reindex(years, fill_value=0.0), axis=1)

sum_of_shares = prop.sum(axis=1).sort_values(ascending=False)
cum_docs = counts.sum(axis=1).sort_values(ascending=False)

top_a = set(sum_of_shares.index[:35])
top_b = set(cum_docs.index[:35])
labels = _load_topic_labels(config)

print("assigned documents in the window: %d over %d years (%d-%d)"
      % (int(counts.values.sum()), len(years), years[0], years[-1]))
print("sum of yearly shares: max %.3f   |   cumulative count: max %d"
      % (sum_of_shares.iloc[0], int(cum_docs.iloc[0])))
print()
print("top-35 sets identical: %s" % (top_a == top_b))
print("in sum-of-shares but not in cumulative-count: %s" % sorted(int(t) for t in top_a - top_b))
print("in cumulative-count but not in sum-of-shares: %s" % sorted(int(t) for t in top_b - top_a))
print()
print("rank comparison for the first 12 topics of each ranking:")
print("  %-4s %-46s | %-4s %-46s" % ("rank", "by sum of yearly shares", "rank", "by cumulative count"))
for i in range(12):
    ta, tb = sum_of_shares.index[i], cum_docs.index[i]
    ra, rb = list(cum_docs.index).index(ta) + 1, list(sum_of_shares.index).index(tb) + 1
    print("  %-4d %-46s | %-4d %-46s"
          % (i + 1, "T%d %s" % (ta, labels.get(int(ta), ""))[:46],
             i + 1, "T%d %s" % (tb, labels.get(int(tb), ""))[:46]))
print()
print("rank shifts > 10 places (sum-of-shares rank vs count rank):")
for t in sum_of_shares.index[:60]:
    ra = list(sum_of_shares.index).index(t) + 1
    rb = list(cum_docs.index).index(t) + 1
    if abs(ra - rb) > 10:
        print("   T%-3d %2d -> %2d   %s" % (t, ra, rb, labels.get(int(t), "")))
