"""Definitive brace/verbatim balance check for both LaTeX documents."""

from __future__ import annotations

from pathlib import Path


def check(path: Path) -> None:
    txt = path.read_text(encoding="utf-8")
    depth = 0
    min_depth = 0
    bad_line = None
    in_comment = False
    for lineno, line in enumerate(txt.splitlines(), start=1):
        # strip an unescaped % comment
        stripped = []
        i = 0
        while i < len(line):
            ch = line[i]
            if ch == "%" and (i == 0 or line[i - 1] != "\\"):
                break
            stripped.append(ch)
            i += 1
        code = "".join(stripped)
        for ch in code:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth < min_depth:
                    min_depth = depth
                    bad_line = lineno
    print(f"{path.name}: final depth {depth}, minimum depth {min_depth}"
          + (f" (first dip at line {bad_line})" if min_depth < 0 else ""))
    # environment balance for the document class
    for env in ("table", "tabular*", "tabular", "figure", "longtable", "verbatim"):
        n_open = txt.count(f"\\begin{{{env}}}")
        n_close = txt.count(f"\\end{{{env}}}")
        if n_open or n_close:
            flag = "" if n_open == n_close else "  <-- MISMATCH"
            print(f"   {env:10s} begin {n_open:3d} / end {n_close:3d}{flag}")


for p in (Path("manuscript/els-cas-templates/manuscript-v4.tex"),
          Path("manuscript/els-cas-templates/manuscript-v3-appendix.tex"),
          Path("manuscript/els-cas-templates/supplementary_tables.tex")):
    if p.exists():
        check(p)
