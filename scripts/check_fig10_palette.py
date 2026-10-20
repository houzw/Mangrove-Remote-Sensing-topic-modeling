"""Assert Fig. 10 draws every displayed topic in its own colour.

The palette is a 16-entry list; the old mapping indexed it by ``tid % 16``, which
gave T36 and T52 the same magenta. Node labels are ``T<id>``, so colour-per-topic
can be read straight off the built figure.
"""

from __future__ import annotations

from collections import Counter

from src.config import get_config
from src.topic_sankey import (
    build_sankey_figure,
    compute_sankey_flows,
    load_similarity_matrix,
    load_topic_labels,
    load_topic_proportions,
)

config = get_config()
sk = config.topic_sankey
flows = compute_sankey_flows(
    load_topic_proportions(config),
    load_similarity_matrix(config),
    load_topic_labels(config),
    temperature=sk.temperature,
    min_flow=sk.min_flow,
    top_n_topics=sk.top_n_topics,
)
figure = build_sankey_figure(flows, config)
node = figure.data[0].node

by_topic: dict[str, str] = {}
for label, colour in zip(node.label, node.color):
    if label in by_topic and by_topic[label] != colour:
        raise SystemExit(f"[FAIL] {label} is drawn in two colours: {by_topic[label]} / {colour}")
    by_topic[label] = colour

print(f"displayed topics : {len(by_topic)}")
for label in sorted(by_topic, key=lambda s: int(s[1:])):
    print(f"  {label:<5s} {by_topic[label]}")

counts = Counter(by_topic.values())
shared = {c: [t for t, cc in by_topic.items() if cc == c] for c, n in counts.items() if n > 1}
print(f"distinct colours : {len(counts)}")
if shared:
    for colour, topics in shared.items():
        print(f"[FAIL] {colour} shared by {topics}")
    raise SystemExit(1)
print("[PASS] every displayed topic has its own colour")
