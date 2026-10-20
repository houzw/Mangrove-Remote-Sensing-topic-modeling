"""Subsection-level integrity check for a proposed edit set.

The per-edit gate (verify_edits.py) asks whether a span keeps its own numbers. That is the
right rule for a sentence that must stand alone, but too strict when an edit legitimately
consolidates a fact that the same subsection states again elsewhere. This check asks the
question that actually matters for the paper: does the SUBSECTION still contain every number,
reference, label and citation key it contained before?

Reports only; never writes.
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
import sys
from collections import Counter

TEX = pathlib.Path("manuscript/els-cas-templates/manuscript-v5.tex")
START = "\\subsection{Spatial Distribution"
END = "\\subsection{Semantic Themes Identified by BERTopic}"


def load(path: str):
    spec = importlib.util.spec_from_file_location("e", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    return getattr(mod, "EDITS", [])


def nums(t: str) -> Counter:
    t = t.replace("{,}", "").replace(",", "")
    return Counter(re.findall(r"\d+(?:\.\d+)?", t))


def cmds(t: str, name: str) -> Counter:
    return Counter(re.findall(r"\\" + name + r"\{([^}]*)\}", t))


def cites(t: str) -> Counter:
    c: Counter = Counter()
    for m in re.finditer(r"\\cite[pt]?\{([^}]*)\}", t):
        for k in m.group(1).split(","):
            if k.strip():
                c[k.strip()] += 1
    return c


def main() -> int:
    txt = TEX.read_text(encoding="utf-8")
    start, end = txt.index(START), txt.index(END)
    seg = txt[start:end]
    print(f"subsection span: {len(seg)} chars, {len(seg.split())} words")

    edits = load(sys.argv[1] if len(sys.argv) > 1 else "scripts_tmp/edits_32.py")
    new = seg
    for i, pair in enumerate(edits, 1):
        old, rep = pair[0], pair[1]
        c = new.count(old)
        print(f"  edit {i}: occurrences in subsection = {c}")
        if c != 1:
            sys.exit(f"anchor {i} is not unique inside the subsection")
        new = new.replace(old, rep, 1)

    print(f"after edits: {len(new.split())} words "
          f"({len(new.split()) - len(seg.split()):+d})")
    print()
    bad = 0
    for label, fn in (("numbers", nums),
                      ("refs", lambda t: cmds(t, "ref")),
                      ("labels", lambda t: cmds(t, "label")),
                      ("cite keys", cites)):
        a, b = fn(seg), fn(new)
        lost = a - b
        print(f"{label:10s} before={sum(a.values()):4d} after={sum(b.values()):4d} "
              f"lost={dict(lost) if lost else 'none'}")
        if lost:
            bad += len(lost)

    brace = (new.count("{") - new.count("}")) - (seg.count("{") - seg.count("}"))
    print(f"brace-balance change: {brace}")
    if brace:
        bad += 1
    print(f"\nRESULT: {'CLEAN' if not bad else f'{bad} PROBLEM(S)'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
