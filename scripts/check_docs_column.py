"""Which definition does the Docs column of tab:emerging-technologies use?

Computes, for the eight candidate fronts, both candidate counts:
  (a) total documents assigned to the topic over all years
  (b) documents assigned in the partial 2026 only
and compares each against the values printed in the manuscript table.
"""

from __future__ import annotations

import pathlib
import re

import pandas as pd

ROOT = pathlib.Path(".")
TEX = ROOT / "manuscript/els-cas-templates/manuscript-v5.tex"
TOPICS = [17, 16, 8, 4, 57, 5, 35, 68]   # the order the author states

# ---- the values actually printed in the table ----
txt = TEX.read_text(encoding="utf-8")
tab = re.search(r"\\begin\{table\}\[pos=[^\]]*\]\s*\\caption\{The eight candidate emerging"
                r".*?\\end\{table\}", txt, re.S).group(0)
printed = {}
for row in tab.split(r"\\")[:-1]:
    m = re.match(r"\s*T(\d+)\s*&(.*)", row)
    if not m:
        continue
    tid = int(m.group(1))
    nums = re.findall(r"(\d+)\s*$", row.rstrip())
    printed[tid] = row
print("=== rows as printed in the manuscript table ===")
for tid in TOPICS:
    row = printed.get(tid, "")
    print(f"  T{tid:<3} {row.strip()[:130]}")

# ---- compute both counts from the assignment table ----
dist = pd.read_csv(ROOT / "data/processed/doc_topic_dist.csv", low_memory=False)
dist["Topic"] = pd.to_numeric(dist["Topic"], errors="coerce")
dist["PY"] = pd.to_numeric(dist["PY"], errors="coerce")
assigned = dist[dist["Topic"] >= 0]
print(f"\nassignment table: {len(dist)} rows, {len(assigned)} topic-assigned, "
      f"years {int(dist['PY'].min())}-{int(dist['PY'].max())}")

total = assigned.groupby("Topic").size()
y2026 = assigned[assigned["PY"] == 2026].groupby("Topic").size()

print(f"\n{'topic':>6} {'(a) all years':>14} {'(b) 2026 only':>14}")
print("-" * 38)
for tid in TOPICS:
    print(f"  T{tid:<3} {int(total.get(tid, 0)):>12} {int(y2026.get(tid, 0)):>14}")

print(f"\nsum over the eight: (a) {sum(int(total.get(t, 0)) for t in TOPICS)}  "
      f"(b) {sum(int(y2026.get(t, 0)) for t in TOPICS)}")

# ---- the author's stated values ----
STATED = {17: 27, 16: 28, 8: 37, 4: 50, 57: 12, 5: 48, 35: 19, 68: 10}
print(f"\n=== comparison against the stated Docs column {STATED} ===")
print(f"{'topic':>6} {'stated':>7} {'(a)':>7} {'(b)':>7}   verdict")
match_a = match_b = 0
for tid in TOPICS:
    s = STATED[tid]
    a = int(total.get(tid, 0))
    b = int(y2026.get(tid, 0))
    tag = []
    if s == a:
        tag.append("=a")
        match_a += 1
    if s == b:
        tag.append("=b")
        match_b += 1
    if not tag:
        tag.append(f"neither (a-b={s - a:+d}, b-a={s - b:+d})")
    print(f"  T{tid:<3} {s:>7} {a:>7} {b:>7}   {', '.join(tag)}")

print(f"\nmatches: (a) {match_a}/8   (b) {match_b}/8")
