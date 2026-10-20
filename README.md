# Mangrove remote sensing topic modeling pipeline — code

It reproduces every number reported in the manuscript from the tables.

Licence: MIT (`LICENSE`).

---

## 1. What is in here

| Path | Contents |
| --- | --- |
| `src/` | 42 pipeline modules — preprocessing, bibliometrics, BERTopic topic modelling, spatial  specialisation, RRMI, burst detection, figure producers |
| `rrmi/` | **standalone RRMI implementation** (`rrMI_indicator.py`) — recomputes the index from a publication list and a resource map with nothing but pandas/numpy |
| `scripts/` | 50 verification scripts (`check_*.py`, `cluster_anchors.py`) that re-derive and assert every number quoted in the text |
| `assets/fig1-framework.drawio` | source of Fig. 1 (rendered by `src/framework_figure.py`) |
| `config.yaml` | all paths, hyperparameters and time windows |
| `requirements.txt` | pinned versions actually used (recommended) |
| `environment.yml` | conda reconstruction (convenience) |
| `pixi.toml`, `pixi.lock` | the original environment, bit-exact |
| `data_pub` | The companion data archive |

---

## 2. Install

```bash
python -m pip install -r requirements.txt
python -m pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
```

Three things to know before you blame the code:

* **numpy is pinned to 1.26.4 on purpose.** numpy 2.x against MKL 2026 breaks `np.dot()` (delay-load failure on `mkl_rt.2.dll`).
* **`gensim` and `spacy==3.7.5` are not in `pixi.toml`** — they were installed afterwards. `gensim` sits on the critical path (topic coherence C_V / NPMI);
  scispacy 0.6.2 cannot load spacy 3.8.x, which is what the lockfile resolves.
  Both are listed explicitly in `requirements.txt`.
* SciBERT (`allenai/scibert_scivocab_uncased`) downloads on first use — the first run needs network access.

Reproducing with pixi instead: `pixi install && pixi run install-scispacy`.

---

## 3. Data you must supply yourself

The pipeline reads three inputs that **cannot** be redistributed:

1. **The WoS corpus.** Retrieved under a Clarivate licence that forbids public redistribution, so it is not in the archive. Two routes:

   * **Ask the authors** — the corresponding author supplies the corpus directly on reasonable request (see `data_pub/README.md` §6). Fastest, and gives byte-identical input.
   * **Re-run the query** yourself — `data_pub/search_query.txt` carries the exact strings and retrieval parameters; drop the tagged-plaintext exports into `data/raw/*.txt`. WoS re-indexes continuously, so a later run will not land on exactly 2,332 records.

   Everything downstream is derived from these; the archive ships *derived* tables only.
2. **HGMF 2020 global mangrove map** (Jia et al. 2023, 10 m) — publicly  available, cite rather than redistribute. Set the local path in
   `config.yaml`:
   ```yaml
   collaboration_network:
     mangrove_shapefile: "D:/HGMF_2020/GlobalMangrove2020/GlobalMangrove2020.shp"
   ```
   A missing shapefile fails loudly at step start by design.
3. **Natural Earth 110m / 50m Admin-0** boundaries — public domain;
   `data/external/ne_*_admin_0_countries/`.

The per-country area table (`data_pub/external/HGMF_country_area.csv`) and the parent-field country counts (`data_pub/external/parent_field_country_counts.xlsx`) are deposited, so the country-level analyses can be reproduced without re-processing the rasters.

---

## 4. Layout expected at run time

Modules resolve paths relative to the **workspace root**, so place this directory beside a `data/` tree:

```
workspace/
├── config.yaml          <- from this repo
├── src/                 <- from this repo
├── scripts/             <- from this repo
├── data/raw/*.txt             (you supply: WoS exports)
├── data/processed/            (pipeline writes here)
├── data/external/             (HGMF area csv, Natural Earth, parent-field counts)
└── outputs/{figures,tables,models}/
```

To verify against the deposited numbers, copy `data_pub/corpus/corpus_metadata.csv` and each `data_pub/derived/**` file into `data/processed/`, and `data_pub/models/bertopic_model/` into `outputs/models/`.

---

## 5. Run order

Modules run as `python -m src.<module>` (the `-m` form keeps `src` importable).

### 5a. Pipeline steps

```bash
python -m src.run_pipeline --step preprocess
python -m src.run_pipeline --step bibliometric
python -m src.run_pipeline --step collaboration   # needs the HGMF shapefile
python -m src.run_pipeline --step topics
python -m src.run_pipeline --step spatial
python -m src.run_pipeline --step evolution
python -m src.run_pipeline --step dynamics
python -m src.run_pipeline --step network
python -m src.run_pipeline --step sankey
python -m src.run_pipeline --step keywords
python -m src.run_pipeline --step combined_viz
python -m src.run_pipeline --step emerging
```

The orchestrator skips steps whose outputs are newer than their inputs; add `--force` to rebuild, `--dry-run` to inspect the plan.

### 5b. Standalone analyses (not wired into the orchestrator)

```bash
python -m src.build_bivariate_attr          # attribute table for the bivariate map
python -m src.area_pub_deficit              # area_share - pub_share deficit
python -m src.rrmi_parent_field             # parent-field RRMI (Tables S7-S8)
python -m src.rrmi_normalization            # all-field / GDP / population alternatives
python -m src.non_mangrove_analysis         # method-export analysis
python -m src.study_area_vs_affiliation     # affiliation vs named study region
python -m src.lq_robustness                 # permutation null + shrinkage
python -m src.multi_indicator_consistency
python -m src.bivariate_choropleth
python -m src.bivariate_choropleth --out-suffix _bluered   # manuscript figure
python -m src.prototype_bootstrap_ci        # skip if topic assignments untouched
python -m src.generate_supplementary_tables # renders supplementary_tables.tex
```

### 5c. Figure producers

```bash
python -m src.framework_figure              # Fig. 1  (from assets/fig1-framework.drawio)
python -m src.country_topic_heatmap         # Fig. 7
python -m src.cluster_evolution_stacked     # Fig. 9
python -m src.topic_sankey                  # Fig. 10  (kaleido PNG needs write access)
python -m src.cluster_dendrogram            # Fig. S1
```

### 5d. The RRMI alone (no pipeline, no BERTopic)

The paper's contribution is the index, so it is also shipped as a self-contained script that needs only pandas and numpy:

```bash
cd rrmi
python rrMI_indicator.py --publications pubs.csv --areas areas.csv --out rrmi.csv
python rrMI_indicator.py --publications pubs.csv --areas areas.csv \
    --min-year 2016 --max-year 2025 --out rrmi_recent.csv   # the 71-country window
```

* `--publications` — one row per publication, with a `countries` column holding
  `;`-separated country names and a `PY` column.
* `--areas` — `country, area_km2`.

Feeding it straight from the deposited archive needs two column renames
(`HGMF_country_area.csv` stores zonal statistics under `Country` / `sum`, and
`corpus_metadata.csv` stores the country list under `CU`):

```bash
python - <<'EOF'
import pandas as pd
p = pd.read_csv("data_pub/corpus/corpus_metadata.csv", encoding="utf-8-sig")
p[["CU", "PY"]].rename(columns={"CU": "countries"}).to_csv("pubs.csv", index=False)
a = pd.read_csv("data_pub/external/HGMF_country_area.csv", encoding="utf-8-sig")
a[["Country", "sum"]].rename(columns={"Country": "country", "sum": "area_km2"}) \
    .to_csv("areas.csv", index=False)
EOF
python rrmi/rrMI_indicator.py --publications pubs.csv --areas areas.csv --out rrmi.csv
```

Verified end-to-end: this returns 76 countries, an area denominator of 143,970.0 km², a fractional-publication denominator of 2,087.17, and China −0.222035 / Indonesia +0.146979 / United States −0.140111 / India −0.09476 — i.e. Table 2 exactly, with Spearman(RRMI, pubs per 10³ km²) = −0.799.

* Default denominators are the **76-country table basis** used in the paper  (143,970 km² of the map's 145,068; 2,087.17 of the corpus's 2,332 fractional publications). `--global-basis` divides by the map and corpus totals instead and moves individual values by up to ~0.02 (China −0.199 vs −0.222) — the  published values are the defaults.
* Territories fold into their metropole on **both** sides; `--no-fold` switches it off.

It reproduces Table 2 exactly: China −0.2220, Indonesia +0.1470, United States −0.1401, India −0.0948. See `rrmi/readme.md`.

### 5e. Verify the manuscript's numbers

```bash
python scripts/check_rrmi_direction.py       # 110x72 matrix, RRMI signs, 76/110 ledger
python scripts/check_rrmi_parent_field.py    # parent-field definitions + orthogonality
python scripts/check_rrmi_normalization.py   # alternative denominators
python scripts/check_china_counts.py         # 570 / 468.2 / 325 disambiguation
python scripts/check_frontier_baseline.py    # burst baseline sparsity
python scripts/check_lifecycle_by_cluster.py # Fig. 11 cluster cross-check
python scripts/check_fig6_heatmap.py
python scripts/check_fig7_legibility.py
python scripts/check_fig9_shares.py
python scripts/check_fig10_sankey.py
python scripts/check_fig10_palette.py
python scripts/check_phase_i_sensors.py
python scripts/check_phase_ii_sensors.py
```

Each script asserts against the artifact's own write precision and prints a pass/fail line; a silent pass is not a pass — read the output.

---

## 6. Reproducibility constraints

* **The topic model is deliberately not re-run.** UMAP is seeded and the  production model is deposited in `data_pub/models/bertopic_model/`. Re-running  `topics` on a different library version will shift topic ids and invalidate every downstream table. To reproduce, restore the deposited model rather than refitting.
* **The vectorizer is pinned to `CountVectorizer`**
  (`config.yaml → topic_modeling.vectorizer.backend: "count"`, `ngram_range=(1,3)`, `max_df=0.85`, `min_df=5`). The production model was built without scispaCy in this path; `backend: "spacy"` switches to
  `KeyphraseCountVectorizer` and is not the published configuration.
* **2026 is partially indexed** (retrieval ended 15 July 2026). It is excluded from trend fitting and from the topic lifecycle; do not read it as a full year.
* **Kaleido PNG export** of Plotly figures can fail in restricted sandboxes; the HTML outputs are always written.
* `src/country_polar_chart.py` is **withheld**: it predates the territory-folding migration, so its per-country numbers are not comparable to the published ones. Do not quote it.

---

## 7. Country-count conventions (the usual source of mismatched numbers)

Three different counts circulate for the same country and they are never interchangeable:

| Name | Meaning | China | Indonesia |
| --- | --- | --- | --- |
| full count | publications credited once per country | 570 | 154 |
| fractional | `1/len(distinct countries)` weighted | 468.2 | 108.3 |
| topic-assigned `N_c` | documents the topic model assigned | 325 | 67.7 |

All per-country derivations route through `src/utils.canonical_countries()` (fold territories → de-duplicate → weight), because folding after weighting double-counts `"China;Taiwan"`.
