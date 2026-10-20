"""Check that every fenced OLD/NEW block in report_34.md matches edits_34.EDITS verbatim."""

from __future__ import annotations

import importlib.util
import pathlib
import re

FENCE = "`" * 3
rep = pathlib.Path("scripts_tmp/report_34.md").read_text(encoding="utf-8")
blocks = re.findall(FENCE + r"\n(.*?)\n" + FENCE, rep, re.S)

spec = importlib.util.spec_from_file_location("e", "scripts_tmp/edits_34.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

print(f"fenced blocks: {len(blocks)}  (expected {2 * len(mod.EDITS)})")
ok = True
for i, (old, new) in enumerate(mod.EDITS, 1):
    for j, want in ((0, old), (1, new)):
        got = blocks[2 * (i - 1) + j]
        if got != want:
            ok = False
            k = 0
            while k < min(len(got), len(want)) and got[k] == want[k]:
                k += 1
            print(f"MISMATCH edit {i} {'OLD' if j == 0 else 'NEW'} at char {k}:")
            print(f"  report: {got[max(0, k - 50):k + 50]!r}")
            print(f"  edits : {want[max(0, k - 50):k + 50]!r}")
print("report blocks match EDITS verbatim:", ok)
ascii_blocks = "".join(blocks)
print("non-ASCII inside quoted blocks:",
      sorted({c for c in ascii_blocks if ord(c) > 127}) or "none")
