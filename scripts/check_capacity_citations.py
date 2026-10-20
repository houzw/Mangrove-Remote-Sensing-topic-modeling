"""Look for a corpus paper that supports the North-South capability claim better than Giri 2011.

Giri et al. (2011) is a global mangrove *distribution map*, not a study of research or
remote-sensing capacity differences. This prints corpus papers whose title/abstract/keywords
mention a capacity, infrastructure or geographic-bias gap, so the sentence can cite one.
"""

from __future__ import annotations

import pandas as pd

d = pd.read_csv("data/processed/cleaned_data.csv", encoding="utf-8-sig")
pat = (r"(?i)(north[- ]south|global south|developing countr|capacity build|research capacity|"
       r"research gap|inequalit|geographic\w* bias|geographical bias|technolog\w* gap|"
       r"infrastructure|under-?represent|collaboration (network|pattern))")
hit = d[d["text"].fillna("").str.contains(pat, regex=True)]
print(f"candidate papers: {len(hit)}")
for _, r in hit.head(14).iterrows():
    print(f"  PY{r['PY']}  {str(r['TI'])[:100]}")
    print(f"        {str(r['SO'])[:70]}")
