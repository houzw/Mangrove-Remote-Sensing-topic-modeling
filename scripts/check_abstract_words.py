"""Throwaway: word/sentence count for an abstract draft.

Usage:
    pixi run python -m scripts_tmp.check_abstract_words            # read from .tex
    pixi run python -m scripts_tmp.check_abstract_words draft.txt  # read from file
"""
import io
import re
import sys

P = "manuscript/els-cas-templates/manuscript-v4.tex"


def clean(t):
    t = t.replace("\\%", "%")
    t = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", "", t)
    t = t.replace("{", "").replace("}", "").replace("$", "").replace("--", "-").replace("~", " ")
    return re.sub(r"\s+", " ", t).strip()


if len(sys.argv) > 1:
    plain = clean(io.open(sys.argv[1], encoding="utf-8").read())
    label = sys.argv[1]
else:
    s = io.open(P, encoding="utf-8", newline="").read()
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", s, re.S)
    plain = clean(m.group(1))
    label = P

words = [w for w in plain.split() if re.search(r"[A-Za-z0-9]", w)]
print("SOURCE:", label)
print("WORDS:", len(words))
sents = [x for x in re.split(r"(?<=[.!?])\s+", plain) if x.strip()]
print("SENTENCES:", len(sents))
for i, x in enumerate(sents, 1):
    print("%2d. (%3dw) %s" % (i, len(x.split()), x))
