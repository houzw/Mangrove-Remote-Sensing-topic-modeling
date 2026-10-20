"""M7 -- inter-annotator agreement on the LLM label-verification step.

    pixi run python -m scripts_tmp.check_label_annotator_agreement

The authors assigned each of the 72 topic labels one of three actions (adopted / reworded /
replaced). An independent model repeated the judgement from the same evidence (keywords plus
the proposed label; the final label was withheld, since it did not exist at that point). This
reports raw agreement and Cohen's kappa for the three-class decision and for the binary
question that matters most to a reader: was the proposed label kept as it stood, or changed?
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(".")
SECOND = ROOT / "data/processed/label_second_annotator.csv"


def kappa(pairs: pd.DataFrame, a: str, b: str) -> tuple:
    n = len(pairs)
    po = (pairs[a] == pairs[b]).mean()
    cats = sorted(set(pairs[a]) | set(pairs[b]))
    pe = sum((pairs[a] == c).mean() * (pairs[b] == c).mean() for c in cats)
    k = (po - pe) / (1 - pe) if pe != 1 else float("nan")
    return po, k, n, cats


def confusion(pairs: pd.DataFrame, a: str, b: str, cats: list) -> None:
    tab = pd.crosstab(pairs[a], pairs[b], dropna=False).reindex(index=cats, columns=cats,
                                                                fill_value=0)
    print(tab.to_string())


def main() -> None:
    d = pd.read_csv(SECOND)
    # The authors' table capitalises the action, the model returns lower case; normalise both
    # before any comparison (an un-normalised crosstab reads as zero agreement).
    for col in ("action", "author_action"):
        d[col] = d[col].astype(str).str.strip().str.capitalize()
    print(f"annotated topics: {len(d)}")
    print(f"author actions : {d['author_action'].value_counts().to_dict()}")
    print(f"model  actions : {d['action'].value_counts().to_dict()}")

    cats = ["Adopted", "Reworded", "Replaced"]
    po, k, n, _ = kappa(d, "author_action", "action")
    print("\nthree-class decision (adopted / reworded / replaced):")
    print(f"  raw agreement {100 * po:.1f}%  Cohen's kappa {k:.3f}  n={n}")
    confusion(d, "author_action", "action", cats)

    d = d.copy()
    d["author_changed"] = d["author_action"] != "Adopted"
    d["model_changed"] = d["action"] != "Adopted"
    po2, k2, n2, _ = kappa(d, "author_changed", "model_changed")
    tp = int((d["author_changed"] & d["model_changed"]).sum())
    fp = int((~d["author_changed"] & d["model_changed"]).sum())
    fn = int((d["author_changed"] & ~d["model_changed"]).sum())
    tn = int((~d["author_changed"] & ~d["model_changed"]).sum())
    print("\nbinary decision (was the proposed label kept as it stood?):")
    print(f"  raw agreement {100 * po2:.1f}%  Cohen's kappa {k2:.3f}  n={n2}")
    print(f"  both kept {tn} | both changed {tp} | model would change, authors kept {fp} | "
          f"authors changed, model kept {fn}")

    if "accurate" in d.columns:
        acc = d["accurate"].astype(str).str.lower().isin(["true", "1"])
        print(f"\nmodel judged the proposed label accurate in {int(acc.sum())} of {len(d)}")
        print(f"  of those, the authors had adopted it: "
              f"{int((acc & ~d['author_changed']).sum())}")
        print(f"  of those, the authors had changed it: {int((acc & d['author_changed']).sum())}")

    print("\nreplaced-label cases (the ones that matter most):")
    for _, r in d[d["author_action"] == "Replaced"].iterrows():
        print(f"  T{r['topic']}: model={r['action']:9s} accurate={r['accurate']}")
        print(f"      proposed {r['proposed']!r} -> final {r['final_label']!r}")


if __name__ == "__main__":
    main()
