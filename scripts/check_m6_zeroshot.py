"""Local zero-shot baseline: can an embedding model beat the 65.5 % gazetteer precision?

    pixi run python -m scripts_tmp.check_m6_zeroshot

Uses the locally cached ``BAAI/bge-m3`` encoder (no network, no API key) to score every
flagged paper against two prompt hypotheses:

  STUDY   "The study area of this paper is <COUNTRY>."
  MENTION "This paper mentions <COUNTRY> only for comparison, context or review."

and compares the resulting decision with the hand-coded labels in
``m6_validation_coded.csv``. This is a *baseline measurement*, not a new pipeline: it
answers "would a semantic scorer do better than the regex gazetteer?" before anyone
pays for an LLM pass.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.study_area_vs_affiliation import LOW_OUTPUT_TROPICAL  # noqa: E402

MODEL = "BAAI/bge-m3"


def main() -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer

    coded = pd.read_csv(ROOT / "data" / "processed" / "m6_validation_coded.csv")
    audit = pd.read_csv(ROOT / "data" / "processed" / "_m6_check_regions.csv")
    audit = audit.drop_duplicates(subset=["DI"], keep="first").set_index("DI")
    coded = coded[coded["DI"].notna()].copy()
    print(f"flagged papers with labels: {len(coded)}")
    print(f"gold: {coded['provisional_label'].value_counts().to_dict()}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModel.from_pretrained(MODEL).to(device).eval()

    def embed(texts: list[str], batch: int = 8) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            enc = tok(chunk, padding=True, truncation=True, max_length=1024,
                      return_tensors="pt").to(device)
            with torch.no_grad():
                h = mdl(**enc).last_hidden_state[:, 0]
            out.append(torch.nn.functional.normalize(h, dim=-1).cpu().numpy())
        return np.vstack(out)

    templates_study = ["The study area of this paper is {c}.",
                       "This paper maps or measures mangroves in {c}."]
    templates_mention = ["The paper mentions {c} only for comparison, background or review.",
                         "{c} is listed among other countries, not studied."]

    rows = []
    for _, r in coded.iterrows():
        row = audit.loc[r["DI"]] if r["DI"] in audit.index else None
        title = str(r.get("TI", ""))
        abstract = str(row.get("AB", "")) if row is not None else ""
        text = (title + " " + abstract)[:6000]
        names = [t for t in str(r.get("regions_abstract", "")).split(";")
                 if t in LOW_OUTPUT_TROPICAL] or ["the study country"]
        country = names[0]
        prompts = ([t.format(c=country.title()) for t in templates_study]
                   + [t.format(c=country.title()) for t in templates_mention])
        pe = embed([text] + prompts)
        doc, pr = pe[0], pe[1:]
        s_study = float(np.max(doc @ pr[:len(templates_study)].T))
        s_ment = float(np.max(doc @ pr[len(templates_study):].T))
        rows.append({"DI": r["DI"], "gold": r["provisional_label"],
                     "s_study": s_study, "s_mention": s_ment,
                     "pred": "yes" if s_study > s_ment else "no"})

    res = pd.DataFrame(rows)
    gold_yes = res["gold"] == "yes"
    tp = int(((res["pred"] == "yes") & gold_yes).sum())
    fp = int(((res["pred"] == "yes") & ~gold_yes).sum())
    fn = int(((res["pred"] == "no") & gold_yes).sum())
    tn = int(((res["pred"] == "no") & ~gold_yes).sum())
    acc = (tp + tn) / len(res)
    print(f"\nconfusion vs hand labels: TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"agreement with the hand coding: {100 * acc:.1f}%")
    print(f"zero-shot precision (yes-calls correct): {100 * tp / max(tp + fp, 1):.1f}%")
    print(f"zero-shot recall                       : {100 * tp / max(tp + fn, 1):.1f}%")
    print("\nbaseline to beat: gazetteer-only precision 65.5 % (gazetteer recall ~100 %)")
    res.to_csv(ROOT / "data" / "processed" / "m6_zeroshot_baseline.csv", index=False,
               encoding="utf-8-sig")
    print("wrote data/processed/m6_zeroshot_baseline.csv")


if __name__ == "__main__":
    main()
