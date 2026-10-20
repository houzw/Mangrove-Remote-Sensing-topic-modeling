"""Analyse the topic-label capitalisation: what would change, and what it costs.

Read-only. Prints, for every topic whose Short_Label is not sentence case:
  * the current and normalised label,
  * the current Action (declared in generate_label_verification.py, else "--"),
  * the Action it would need afterwards ("Reworded" whenever the string changes),
and the resulting adopted/reworded/replaced counts.

Normalisation = sentence case, keeping the first word, acronyms (AGB, SAR, UAV, ...) and the
proper nouns that actually occur (India, Southeast, Asia).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pandas as pd

KEEP = {"AGB", "SAR", "UAV", "LiDAR", "India", "Southeast", "Asia", "ICESat-2", "InSAR", "PolSAR"}


def _word(w: str) -> str:
    if "-" in w:
        head, *tail = w.split("-")
        head_keep = head.isupper() or head in KEEP
        head = head if head_keep else head.lower()
        return "-".join([head] + [_word(t) for t in tail])
    if w in KEEP or (w.isupper() and len(w) >= 2) or re.search(r"\d", w):
        return w
    return w.lower()


def normalise(label: str) -> str:
    words = str(label).split()
    if not words:
        return label
    # the first word is normalised too (so "UAV-Based" -> "UAV-based"), then re-capitalised
    first = _word(words[0])
    first = first if first[:1].isupper() else first[:1].upper() + first[1:]
    out = [first]
    out += [_word(w) for w in words[1:]]
    return " ".join(out)


# --- current Action, declared in the verification generator -------------------
gen = Path("scripts_tmp/generate_label_verification.py").read_text(encoding="utf-8")
block = gen[gen.index("FINAL_LABELS = {"): gen.index("}", gen.index("FINAL_LABELS = {"))]
declared: dict[int, tuple[str, str]] = {}
for line in block.splitlines():
    m = re.match(r"\s*(\d+):\s*\((.*)\)\s*,?\s*$", line)
    if not m:
        continue
    try:
        label, action = ast.literal_eval("(" + m.group(2) + ")")
    except Exception:
        continue
    declared[int(m.group(1))] = (label, action)

t = pd.read_csv("data/processed/topics_info.csv", encoding="utf-8-sig")
t = t[t.Topic != -1].sort_values("Topic")

def action_of(tid: int, label: str, llm: str) -> str:
    # Topics the generator does not declare keep the pipeline label verbatim -> "--";
    # the generator does NOT compare against the LLM column (whose cell is a list literal).
    return declared[tid][1] if tid in declared else "--"

changes = []
for _, r in t.iterrows():
    tid = int(r.Topic)
    cur = str(r.Short_Label).strip()
    new = normalise(cur)
    if new != cur:
        llm = str(r.get("LLM", "")).strip()
        changes.append((tid, cur, new, action_of(tid, cur, llm)))

print(f"labels that would change: {len(changes)} of {len(t)}\n")
print(f"{'topic':<7}{'action now':<12}{'current label':<48}normalised")
for tid, cur, new, act in changes:
    print(f"  T{tid:<5d}{act:<12}{cur:<48}{new}")

before = {"--": 0, "Reworded": 0, "Replaced": 0}
after = {"--": 0, "Reworded": 0, "Replaced": 0}
changed = {tid for tid, *_ in changes}
for _, r in t.iterrows():
    tid = int(r.Topic)
    llm = str(r.get("LLM", "")).strip()
    a = action_of(tid, str(r.Short_Label).strip(), llm)
    before[a] += 1
    a2 = "Reworded" if (tid in changed and a == "--") else a
    after[a2] += 1

print(f"\nadopted/reworded/replaced  now: {before['--']}/{before['Reworded']}/{before['Replaced']}")
print(f"                           after: {after['--']}/{after['Reworded']}/{after['Replaced']}")
print("\ntopics whose Action would flip to Reworded:",
      sorted(tid for tid, *_ in changes if action_of(tid, "", "") == "--"))
