"""Verify the size-adjusted RRMI artifact and every number the manuscript quotes.

Checks, from the artifacts rather than from the prose:

  P1  algebra: the decomposition identity row by row, and that each share sums to 1
  P2  the frozen set: the RRMI top-14 in ``rrmi_normalization.csv`` is the same
      set as the top-14 of ``area_pub_deficit.csv`` (a drift raises)
  P3  the manual Spearman implementation against two already-published
      correlations (RRMI vs publication density -0.80; RRMI vs citation gap +0.94),
      so the new rho values rest on a validated estimator
  P4  the headline claims: RCA > 1 and capacity > 0 for all 14; the capacity and
      field-specific sums; the alternative-denominator correlations and overlaps
  P5  no stale LaTeX fragment (src.rrmi_parent_field owns Tables S7--S8)

Run: pixi run python scripts_tmp/check_rrmi_normalization.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rrmi_normalization import spearman  # noqa: E402

PROC = Path("data/processed")
TEX = Path("manuscript/els-cas-templates/rrmi_normalization_table.tex")
TOL = 2e-6

failures: list[str] = []


def check(condition: bool, label: str, detail: str = "") -> None:
    status = "OK  " if condition else "FAIL"
    print(f"  [{status}] {label}{(' -- ' + detail) if detail else ''}")
    if not condition:
        failures.append(label)


base = pd.read_csv(PROC / "rrmi_normalization.csv", encoding="utf-8-sig")
raw = pd.read_csv(PROC / "area_pub_deficit.csv", encoding="utf-8-sig")
summary = json.loads((PROC / "rrmi_normalization_summary.json").read_text(encoding="utf-8"))

print("=" * 78)
print("P1  algebra")
print("=" * 78)
resid = (base["capacity_term"] + base["field_term"] - base["rrmi"]).abs().max()
check(resid < TOL, "decomposition identity, all rows", f"max residual {resid:.2e}")
check(abs(base["area_share"].sum() - 1) < 1e-4, "area shares sum to 1",
      f"{base['area_share'].sum():.6f}")
check(abs(base["pub_share"].sum() - 1) < 1e-4, "publication shares sum to 1",
      f"{base['pub_share'].sum():.6f}")
check(abs(base["articles_share"].sum() - 1) < 1e-4, "all-field shares sum to 1 over the 76",
      f"{base['articles_share'].sum():.6f}")
check(base["articles"].notna().all(), "all-field data present for every row",
      f"{int(base['articles'].notna().sum())} of {len(base)}")
merged = base.merge(raw, on="country", suffixes=("_norm", "_raw"))
check(len(merged) == len(base), "row sets agree with area_pub_deficit.csv", f"{len(merged)} rows")
resid_rrmi = (merged["rrmi"] - merged["deficit"]).abs().max()
check(resid_rrmi < 1e-6, "headline RRMI reproduced", f"{resid_rrmi:.2e}")

print()
print("=" * 78)
print("P2  the frozen top-14")
print("=" * 78)
top14_new = list(base.sort_values("rrmi", ascending=False)["country"].head(14))
top14_raw = list(raw.sort_values("deficit", ascending=False)["country"].head(14))
check(top14_new == top14_raw, "top-14 set and order unchanged", ", ".join(top14_new[:4]) + ", ...")

print()
print("=" * 78)
print("P3  the Spearman implementation (against published values)")
print("=" * 78)
multi = pd.read_csv(PROC / "multi_indicator_consistency.csv", encoding="utf-8-sig")
rho_density = spearman(multi["deficit"], multi["pub_density"])
rho_citation = spearman(multi["deficit"], multi["citation_gap"])
check(abs(rho_density - (-0.80)) < 0.005, "RRMI vs publication density is -0.80",
      f"recomputed {rho_density:+.4f}")
check(abs(rho_citation - 0.94) < 0.005, "RRMI vs citation gap is +0.94",
      f"recomputed {rho_citation:+.4f}")

print()
print("=" * 78)
print("P4  the claims")
print("=" * 78)
t = base[base["country"].isin(top14_new)]
check(int((t["rca"] > 1).sum()) == 14, "all 14 have RCA > 1",
      f"min {t['rca'].min():.2f}, median {t['rca'].median():.2f}")
check(int((t["capacity_term"] > 0).sum()) == 14 and int((t["field_term"] > 0).sum()) == 0,
      "capacity term positive and field term negative for all 14",
      f"sum a-s {t['capacity_term'].sum():+.4f}, sum s-p {t['field_term'].sum():+.4f}")
check(abs(t["area_share"].sum() - 0.635) < 0.001 and abs(t["pub_share"].sum() - 0.167) < 0.001,
      "top-14 hold 63.5% of area and 16.7% of publications",
      f"{100 * t['area_share'].sum():.1f}% / {100 * t['pub_share'].sum():.1f}%")
check(abs(100 * summary["articles_world_share_of_table"] - 71.2) < 0.1,
      "table holds 71.2% of world article output",
      f"{100 * summary['articles_world_share_of_table']:.1f}%")
check(int((base["rca"] < 1).sum()) == 12, "12 countries have RCA < 1",
      f"{int((base['rca'] < 1).sum())} countries; lowest {base['rca'].min():.2f}")
check(summary["coverage"]["rd"] == 57, "R&D coverage 57 of 76 (why it is not a denominator)",
      f"{summary['coverage']['rd']} of {summary['n_countries']}")
for key, rho, ovl in (("articles_latest", 0.950, 11), ("gdp", 0.945, 12), ("population", 0.909, 12)):
    c = summary["comparisons"][key]
    check(abs(c["spearman_vs_rrmi"] - rho) < 0.001 and c["top14_overlap"] == ovl,
          f"{key}: rho {rho:.3f}, top-14 overlap {ovl}",
          f"rho {c['spearman_vs_rrmi']:.4f}, overlap {c['top14_overlap']}, "
          f"jaccard {c['top14_jaccard']:.3f}, in {c['top14_only_in_variant']}")
for country, rca in (("Nigeria", 1.37), ("China", 0.59), ("United States", 0.79),
                     ("Papua New Guinea", 20.27), ("Guinea", 2.55)):
    row = base.set_index("country").loc[country]
    check(abs(row["rca"] - rca) < 0.01, f"{country} RCA {rca:.2f}", f"{row['rca']:.2f}")

print()
print("=" * 78)
print("P5  no duplicate supplementary tables")
print("=" * 78)
check(not TEX.exists(), "no stale S7--S8 fragment (src.rrmi_parent_field owns them)",
      f"{TEX.name} present" if TEX.exists() else "absent")

print()
print("=" * 78)
if failures:
    print(f"FAILED {len(failures)} check(s): " + "; ".join(failures))
    sys.exit(1)
print("All checks passed.")
