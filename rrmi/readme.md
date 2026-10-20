# RRMI — research–resource mismatch index

Companion code for

> **A research–resource mismatch index for mangrove remote sensing: topic
> evolution, emerging fronts, and who leads the science**
> Z.-W. Hou, J. Sun, L. Niu, Z. Li, Q. Sun, Z. Cai, D. Pan
> (submitted)

The paper's contribution is an **indicator**, so this repository is organised
around making that indicator recomputable and auditable. The bibliometric
analysis is the demonstration; the index is the deliverable.

---

## 1. The indicator in one line

```
RRMI_c = a_c − p_c
```

| Symbol | Meaning | Counting convention |
|---|---|---|
| `a_c` | country *c*'s share of the ecological resource (global mangrove area, km²) | share of the **analysis table**, i.e. normalised over the countries that hold both an area row and at least one publication (76 here; `--global-basis` switches to the whole map) |
| `p_c` | country *c*'s share of **fractional** publications | fractional (`1/m_d`), normalised over the same table (`--global-basis` switches to the whole corpus); full-count integers are reported but never enter the index |

**Denominators.** Both shares default to the *table* basis used in the paper:
the 76 countries with both data types hold 143,970 of the map's 145,068 km² and
2,087.17 of the corpus's 2,332 fractional publications, and `a_c` and `p_c` sum
to 1 over that table. `--global-basis` divides by the map's and the corpus's own
totals instead; the two bases move individual RRMI values by up to ~0.02
(China −0.199 vs −0.222) and can reshuffle ranks near the middle of the table,
so the published values should be reproduced with the default.

**Territories.** Administered territories are folded into their metropole on
*both* sides — the area rows and the publication rows alike — so a country's
resource and its research are measured on the same footing (French Guiana's
mangroves and its papers both count towards France; without area-side folding
France would enter the index with 0 km² and drop out). Taiwan is folded on the
publication side only, because the resource map has no Taiwan row. The full
fold table, the HGMF spelling corrections (`Camerocn`→`Cameroon`,
`Venzuela`→`Venezuela`, `Congo.DRC`→`Congo`, …) and the three excluded
affiliation false positives (`Niger`, `Georgia`, `Mongolia`) are listed at the
top of `rrMI_indicator.py` and mirror `src/area_pub_deficit.py` exactly.

* `RRMI_c ∈ [−1, 1]`; positive = under-publishing relative to area, negative = over-publishing.
* Because it is a **difference** of shares rather than a ratio, low output cannot
  inflate it without bound — a country with almost no output simply scores close
  to its full area share. Read the positive extreme together with absolute
  publication counts.
* Computed on every country with **both** an area row and at least one
  publication: **no publication-volume floor**.
* Partition-free: it needs no topic model, only a resource map and a
  publication list.

## 2. Quick start — recompute the index

```bash
pip install pandas numpy scipy          # or: pip install -r requirements.txt
python rrMI_indicator.py \
    --publications data/publications.csv \
    --areas        data/mangrove_area.csv \
    --out          outputs/rrmi.csv
```

Input schemas:

* `publications.csv` — one row per publication; a `countries` column holding
  affiliation countries separated by `;`; optional `year`.
* `mangrove_area.csv` — `country`, `area_km2`.

Useful flags:

| Flag | Effect |
|---|---|
| `--min-year` / `--max-year` | restrict the publication window (used for the 2016–2025 robustness check, Spearman ρ = 0.99 against the headline ranking) |
| `--no-fold` | keep administered territories separate instead of folding them into their metropole (on both sides) |
| `--global-basis` | normalise `a_c` and `p_c` over the whole map and the whole corpus instead of over the analysis table — *not* the published convention |

The script also prints `Spearman(RRMI, publications per 10³ km²)`, the
ratio-form diagnostic reported in the paper (ρ = −0.80, *n* = 76).

## 3. The Web of Science query

Retrieved 15 July 2026 from the WoS Core Collection, January 1972 – July 2026,
peer-reviewed articles, proceedings and reviews in English:

```
TS=(("mangrove*" OR "mangal" OR "rhizophoraceae") AND ("remote sensing" OR "earth observation" OR "satellite imagery" OR "aerial photography" OR "airborne" OR "UAV" OR "UAS" OR "drone" OR "LiDAR" OR "SAR" OR "synthetic aperture radar" OR "optical remote sensing" OR "multispectral" OR "hyperspectral" OR "radar" OR "InSAR" OR "PolSAR" OR "RGB"))
```

The parent-field normalisation (Table S7/S8 of the supplementary PDF) uses the
**same** string with the mangrove block removed.

## 4. Reproducing the demonstration

The full pipeline lives in the project's `src/` package and is imported as a
module (`src.config`, `src.utils` provide paths and logging). Run order:

| Step | Module | Output |
|---|---|---|
| 1 | `src/merge_txt_to_csv.py` | merge WoS tagged exports into one CSV |
| 2 | `src/data_preprocessing.py` | deduplicated corpus, country extraction |
| 3 | `src/bibliometric_analysis.py`, `src/logistic_growth_ci.py` | annual series, logistic fit (Fig. 2) |
| 4 | `src/topic_clusters.py` (SciBERT → UMAP → HDBSCAN → c-TF-IDF + LLM labels) | 72 topics, 6 clusters (Table 3) |
| 5 | `src/topic_dynamics.py`, `src/tech_evolution_stacked.py` | phase evolution (Fig. 8) |
| 6 | `src/burst_detection_kleinberg.py`, `src/emerging_tech_detection.py` | 8 candidate fronts (Table 5, Fig. 7) |
| 7 | `src/area_pub_deficit.py` | **RRMI** (Table 2, Fig. 3) |
| 8 | `src/rrmi_parent_field.py` | parent-field normalisation (Tables S7–S8) |
| 9 | `src/spatial_topic_analysis.py`, `src/lq_robustness.py` | Location Quotient + shrinkage (Table 4, Fig. 6) |
| 10 | `src/generate_supplementary_tables.py` | supplementary tables S1–S4 |

```bash
python -m src.run_pipeline        # steps 2-9 end to end
```

`rrMI_indicator.py` in this folder is a dependency-free re-implementation of
step 7 for readers who only want the indicator.

## 5. Reproducibility notes

* **Random seed.** UMAP is seeded (`random_state=42`); HDBSCAN is deterministic
  given the projection. The burst automaton uses ten seeded restarts.
* **Topic count** is bounded within 60–83 across parameter scans; the RRMI is
  unaffected because it never touches the topic partition.
* **Area snapshot.** HGMF 2020 (Jia et al., 2023). It is held fixed against a
  1972–2026 publication window, so the index is a cross-sectional contrast.
* **Known biases.** Partial 2026 indexing and English-language
  over-representation truncate the corpus toward established producers and
  therefore bias the reported gap toward **understatement**.

## 6. Licence and citation

MIT. If you use the index, please cite the paper; if you use the pipeline,
please also cite BERTopic (Grootendorst, 2022) and the HGMF mangrove map
(Jia et al., 2023).
