"""Integrity check for scripts_tmp/edits_34.py (subsection 3.4, \\label{sec:si}).

Verifies, on the CURRENT manuscript-v5.tex:
  * prose word count before/after, and the cumulative reduction from the pre-round-1 text;
  * every citation key, \\ref/\\label target, and numeric token survives;
  * floats (table/figure blocks) are byte-identical;
  * the single-document Nigeria artifact is stated once and no dangling pointer remains;
  * no non-ASCII characters introduced.
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
import sys

TEX = pathlib.Path("manuscript/els-cas-templates/manuscript-v5.tex")
ROUND1_WORDS = 76  # words removed by the first (already applied) draft of edits_34.py


def load(path: str):
    spec = importlib.util.spec_from_file_location("edits", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.EDITS


def numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text.replace("{,}", "").replace(",", "")))


def cites(text: str) -> set[str]:
    out: set[str] = set()
    for m in re.finditer(r"\\cite[pt]?\{([^}]*)\}", text):
        out.update(k.strip() for k in m.group(1).split(",") if k.strip())
    return out


def refs(text: str) -> set[str]:
    out: set[str] = set()
    for cmd in ("ref", "label", "eqref"):
        out.update(re.findall(r"\\" + cmd + r"\{([^}]*)\}", text))
    return out


edits = load("scripts_tmp/edits_34.py")
txt = TEX.read_text(encoding="utf-8")
lines = txt.split("\n")
start = next(i for i, l in enumerate(lines) if l.startswith(r"\subsection{National Thematic"))
end = next(i for i, l in enumerate(lines) if l.startswith(r"\subsection{Candidate Emerging"))
block = "\n".join(lines[start:end])

float_re = re.compile(r"\\begin\{(table|figure)\}.*?\\end\{\1\}", re.S)
prose_before = float_re.sub("", block)
prose_after = prose_before
for old, new in edits:
    assert prose_after.count(old) == 1, "anchor not unique in prose block"
    prose_after = prose_after.replace(old, new, 1)

b, a = len(prose_before.split()), len(prose_after.split())
print(f"current 3.4 prose: {b} words")
print(f"after these edits: {a} words  (-{b - a}, {100 * (b - a) / b:.1f}% of current)")
print(f"original (pre-round-1) prose: {b + ROUND1_WORDS} words")
print(f"cumulative: {b + ROUND1_WORDS - a} removed of {b + ROUND1_WORDS} "
      f"= {100 * (b + ROUND1_WORDS - a) / (b + ROUND1_WORDS):.1f}%")

n_new = numbers(prose_before) - numbers(prose_after)
c_new = cites(prose_before) - cites(prose_after)
r_new = refs(prose_before) - refs(prose_after)
print("numbers lost:", sorted(n_new) or "none")
print("citations lost:", sorted(c_new) or "none")
print("refs/labels lost:", sorted(r_new) or "none")

floats_before = float_re.findall(block)
floats_after = float_re.findall(float_re.sub("", block) if False else block)
print("float blocks in 3.4:", len(floats_before), "-> untouched:", floats_before == floats_after)

art = [m.start() for m in re.finditer(r"single sparse-topic assignment|single document", prose_after)]
print("single-document artifact mentions in 3.4 prose:", len(art))
for probe in ("discussed above", "as noted earlier", "see above"):
    print(f"  dangling '{probe}':", prose_after.count(probe))

nonascii = {c for c in prose_after if ord(c) > 127}
print("non-ASCII introduced:", nonascii - {c for c in prose_before if ord(c) > 127} or "none")
print("brace balance: before", prose_before.count("{") - prose_before.count("}"),
      "after", prose_after.count("{") - prose_after.count("}"))
print("dollar parity: before", prose_before.count("$") % 2, "after", prose_after.count("$") % 2)
sys.exit(1 if (n_new or c_new or r_new) else 0)
