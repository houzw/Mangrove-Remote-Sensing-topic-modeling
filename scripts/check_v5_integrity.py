"""Verify v5 kept every citation key, label and reference after the compression passes."""

from __future__ import annotations

import pathlib
import re

SRC = pathlib.Path("manuscript/els-cas-templates")
V4 = SRC / "manuscript-v4.tex"


def survey(path: pathlib.Path) -> dict:
    t = path.read_text(encoding="utf-8")
    keys: set[str] = set()
    for m in re.finditer(r"\\cite[pt]?\{([^}]*)\}", t):
        keys.update(k.strip() for k in m.group(1).split(",") if k.strip())
    return {
        "chars": len(t),
        "citations": keys,
        "labels": set(re.findall(r"\\label\{([^}]*)\}", t)),
        "refs": set(re.findall(r"\\ref\{([^}]*)\}", t)),
    }


for name in ("manuscript-v5.tex", "manuscript-v5-diff.tex"):
    p = SRC / name
    s = survey(p)
    print(f"{name}: {s['chars']} chars, {len(s['citations'])} keys, "
          f"{len(s['labels'])} labels, {len(s['refs'])} refs")

v4, v5 = survey(V4), survey(SRC / "manuscript-v5.tex")
print()
print(f"citation keys: v4 {len(v4['citations'])} -> v5 {len(v5['citations'])}")
lost = v4["citations"] - v5["citations"]
added = v5["citations"] - v4["citations"]
print(f"  lost : {sorted(lost) if lost else 'none'}")
print(f"  added: {sorted(added) if added else 'none'}")

print(f"labels: v4 {len(v4['labels'])} -> v5 {len(v5['labels'])}")
print(f"  lost: {sorted(v4['labels'] - v5['labels']) or 'none'}")

# every \ref target must exist as a label, or be an intentional cross-document pointer
unresolved = {r for r in v5["refs"] if r not in v5["labels"]}
print(f"refs with no label in v5: {sorted(unresolved) if unresolved else 'none'}")

app = (SRC / "manuscript-v5-appendix.tex").read_text(encoding="utf-8")
for r in sorted(unresolved):
    print(f"  {r}: in appendix? {r in app}")
