# -*- coding: utf-8 -*-
"""Independent check of the subsection-3.3 compression edits (no manuscript write)."""
import hashlib
import pathlib
import re
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path("scripts_tmp").resolve()))
import edits_33 as E  # noqa: E402

MS = pathlib.Path("manuscript/els-cas-templates/manuscript-v5.tex")
raw = MS.read_bytes()
t = raw.decode("utf-8")
print("manuscript sha1:", hashlib.sha1(raw).hexdigest())
print("has NUL bytes:", b"\x00" in raw)
print("line count:", len(t.splitlines()))

new = t
for i, (old, rep) in enumerate(E.EDITS, 1):
    assert new.count(old) == 1, f"anchor {i} not unique"
    new = new.replace(old, rep, 1)

lines, nlines = t.splitlines(), new.splitlines()
start = next(i for i, l in enumerate(lines) if r"\label{sec:themes}" in l)
end = next(i for i, l in enumerate(lines)
           if r"\label{sec:si}" in l and l.lstrip().startswith(r"\subsection"))
assert start < end, (start, end)
print(f"subsection lines: {start + 1}..{end} (1-based, exclusive end)")

old_sub = "\n".join(lines[start:end])
new_sub = "\n".join(nlines[start:end])
assert old_sub and new_sub


def feats(s):
    return {
        "numbers": Counter(re.findall(r"\d[\d,]*(?:\.\d+)?", s)),
        "topics": Counter(re.findall(r"T\d+", s)),
        "clusters": Counter(re.findall(r"Cluster~[A-F]", s)),
        "cluster_names": Counter(re.findall(r"Cluster [A-F]: [A-Za-z, \-]+", s)),
        "cites": Counter(re.findall(r"\\cite[tp]\{([^}]*)\}", s)),
        "refs": Counter(re.findall(r"\\ref\{([^}]*)\}", s)),
        "labels": Counter(re.findall(r"\\label\{([^}]*)\}", s)),
        "Srefs": Counter(re.findall(r"\\S\\ref\{([^}]*)\}", s)),
        "cmds": Counter(re.findall(r"\\[a-zA-Z]+", s)),
    }


a, b = feats(old_sub), feats(new_sub)
ok = True
for k in a:
    if a[k] != b[k]:
        ok = False
        print(f"MISMATCH {k}")
        print("  lost:", list((a[k] - b[k]).elements()))
        print("  added:", list((b[k] - a[k]).elements()))
    else:
        print(f"ok  {k}: {len(a[k])} distinct tokens preserved")

claim = "no near-duplicate pair to merge"
print("claim present:", claim in new_sub,
      "| 2{,}556:", "2{,}556" in new_sub, "| 0.499:", "0.499" in new_sub,
      "| 0.5 bound:", "no pair exceeds $0.5$" in new_sub)

for name, s in (("old", old_sub), ("new", new_sub)):
    print(f"{name}: non-ascii={sorted({c for c in s if ord(c) > 127})} "
          f"$balance={s.count('$') % 2 == 0} "
          f"braces={s.count('{') == s.count('}')} "
          f"desc={s.count(chr(92) + 'begin{description}')}/{s.count(chr(92) + 'end{description}')}")

env_old = [l for l in lines[start:end]
           if l.lstrip().startswith(("\\begin{table}", "\\begin{figure}", "\\item[", "\\caption"))]
env_new = [l for l in nlines[start:end]
           if l.lstrip().startswith(("\\begin{table}", "\\begin{figure}", "\\item[", "\\caption"))]
print("item/caption/env lines identical:", env_old == env_new)
print("table+figure block bytes identical:",
      "\n".join(lines[start:end])[("\n".join(lines[start:end])).find(r"\begin{table}"):]
      == "\n".join(nlines[start:end])[("\n".join(nlines[start:end])).find(r"\begin{table}"):])


def prose(s):
    keep, skip = [], False
    for l in s.splitlines():
        st = l.strip()
        if st.startswith((r"\begin{table}", r"\begin{figure}")):
            skip = True
        if not skip:
            keep.append(l)
        if st.startswith((r"\end{table}", r"\end{figure}")):
            skip = False
    return " ".join(keep)


print("subsection prose words:", len(prose(old_sub).split()), "->", len(prose(new_sub).split()))
print("ALL CHECKS PASS" if ok else "FAILURES PRESENT")
