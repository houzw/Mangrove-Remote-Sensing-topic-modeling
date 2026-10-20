"""Assert the six cluster names are one canonical set across every artifact.

Sources compared:
  * ``src/topic_clusters.py``  -- ``CLUSTERS`` (the taxonomy; also drives the Fig. 9 and
    Fig. S1 legends, which print ``f"{letter}: {name}"`` at plot time)
  * ``manuscript-v3.tex``      -- the Sec. 3.3 cluster bullets and the Table 3 rows
  * ``label_verification_appendix.tex`` -- the Table S1 caption's name list (lowercase)

Exit status 1 on any mismatch or missing row.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLUSTERS_PY = ROOT / "src" / "topic_clusters.py"
TEX = ROOT / "manuscript" / "els-cas-templates" / "manuscript-v3.tex"
S1 = ROOT / "manuscript" / "els-cas-templates" / "label_verification_appendix.tex"

LETTERS = "ABCDEF"


def norm(s: str) -> str:
    s = s.replace(r"\&", "&").replace("~", " ")
    return re.sub(r"\s+", " ", s).strip().rstrip(".").lower()


def from_code() -> dict[str, str]:
    text = CLUSTERS_PY.read_text(encoding="utf-8")
    pairs = re.findall(r'"letter":\s*"([A-F])",\s*\n\s*"name":\s*"([^"]+)"', text)
    return dict(pairs)


def from_bullets() -> dict[str, str]:
    text = TEX.read_text(encoding="utf-8")
    return dict(re.findall(r"\\item\[Cluster ([A-F]): ([^\]]+)\]", text))


def from_table() -> dict[str, str]:
    text = TEX.read_text(encoding="utf-8")
    rows = re.findall(r"^([A-F]): (.+?) & T\d+,", text, flags=re.MULTILINE)
    return dict(rows)


def from_s1_caption() -> dict[str, str]:
    text = S1.read_text(encoding="utf-8")
    m = re.search(r"thematic clusters of Table~3 of the main text \((.*?)\), so that", text, re.S)
    if not m:
        raise SystemExit("[FAIL] Table S1 caption has no cluster name list")
    body = re.sub(r"\s+", " ", m.group(1))
    return {letter: name for letter, name in re.findall(r"([A-F]), ([^;]+)", body)}


def main() -> None:
    sources = {
        "code CLUSTERS": from_code(),
        "Sec. 3.3 bullets": from_bullets(),
        "Table 3": from_table(),
        "Table S1 caption": from_s1_caption(),
    }
    for label, mapping in sources.items():
        missing = [x for x in LETTERS if x not in mapping]
        if missing:
            raise SystemExit(f"[FAIL] {label}: missing {missing}")

    ref = sources["code CLUSTERS"]
    width = max(len(v) for v in ref.values())
    print(f"{'':22}{'A-F':<6}{'name':<{width}}")
    ok = True
    for letter in LETTERS:
        print(f"  {letter}: {ref[letter]}")
    for label, mapping in sources.items():
        for letter in LETTERS:
            if norm(mapping[letter]) != norm(ref[letter]):
                ok = False
                print(f"[MISMATCH] {label} {letter}: {mapping[letter]!r} != {ref[letter]!r}")
    print("\n" + ("[PASS] all four sources agree on all six cluster names"
                  if ok else "[FAIL] cluster names differ"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
