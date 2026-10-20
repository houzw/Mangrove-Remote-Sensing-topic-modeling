"""Measure (and later re-measure) the six cluster descriptions in §3.x.

Prints words / characters / sentences per cluster bullet so the six can be made
comparable in length, and flags the quantitative anchors each one carries.
"""

import re
from pathlib import Path

tex = Path("manuscript/els-cas-templates/manuscript-v3.tex").read_text(encoding="utf-8")
start = tex.index(r"\subsection{Semantic Themes Identified by BERTopic}")
end = tex.index(r"\end{description}", start)
block = tex[start:end]

items = re.findall(r"\\item\[Cluster ([A-F])[^\]]*\]\s*(.*)", block)
print("%-3s %6s %6s %5s  %s" % ("cl", "words", "chars", "sent", "quantitative anchors"))
for letter, body in items:
    body = body.strip()
    words = len(body.split())
    sents = [s for s in re.split(r"(?<=[.;])\s+", body) if s.strip()]
    nums = re.findall(r"\d[\d,\.]*\s*(?:%|\\%)?|T\d+", body)
    print("%-3s %6d %6d %5d  %s" % (letter, words, len(body), len(sents), " ".join(nums[:14])))

lengths = [len(b.split()) for _, b in items]
if lengths:
    print()
    print("words min/max/mean: %d / %d / %.1f   spread = %.1fx"
          % (min(lengths), max(lengths), sum(lengths) / len(lengths), max(lengths) / min(lengths)))
