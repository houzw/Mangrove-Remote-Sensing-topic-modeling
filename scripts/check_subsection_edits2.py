"""Corrected subsection check: distinguish a REDUCED count from a FULLY ABSENT number.

Counter subtraction (a - b) reports how many occurrences disappeared, so a number going from
two mentions to one shows up as "lost: 1" even though the paper still states it. Only an
empty after-set is real loss.
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


def main() -> int:
    txt = TEX.read_text(encoding="utf-8")
    seg = txt[txt.index(START):txt.index(END)]
    edits = load(sys.argv[1] if len(sys.argv) > 1 else "scripts_tmp/edits_32.py")

    new = seg
    for pair in edits:
        new = new.replace(pair[0], pair[1], 1)

    print(f"{len(seg.split())} -> {len(new.split())} words\n")
    bad = 0
    for label, fn in (("numbers", nums), ("refs", lambda t: cmds(t, "ref")),
                      ("labels", lambda t: cmds(t, "label"))):
        a, b = fn(seg), fn(new)
        absent = sorted(set(a) - set(b))
        reduced = {k: (a[k], b[k]) for k in a if b[k] < a[k]}
        print(f"{label}: distinct before={len(a)} after={len(b)}")
        print(f"   FULLY ABSENT : {absent if absent else 'none'}")
        print(f"   count reduced: {reduced if reduced else 'none'}")
        bad += len(absent)

    print(f"\nRESULT: {'SAFE - nothing is fully absent' if not bad else f'{bad} NUMBER(S) FULLY LOST'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
