"""Assert every \\cite key in the manuscript exists in the .bib, and report the count."""
import io
import re

TEX = "manuscript/els-cas-templates/manuscript-v4.tex"
BIB = "manuscript/els-cas-templates/mangrove-rs-biblio.bib"

tex = io.open(TEX, encoding="utf-8", newline="").read()
bib = io.open(BIB, encoding="utf-8", newline="").read()

have = set(re.findall(r"@\w+\{([^,]+),", bib))
cited = set()
for g in re.findall(r"\\cite[a-z]*\{([^}]*)\}", tex):
    cited |= {k.strip() for k in g.split(",") if k.strip()}

miss = sorted(cited - have)
print("bib entries  :", len(have))
print("cited unique :", len(cited))
print("MISSING KEYS :", miss if miss else "none")
print("uncited bib  :", sorted(have - cited))
