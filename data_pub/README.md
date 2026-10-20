# Mangrove remote sensing bibliometric corpus — derived data

Licence: CC BY 4.0 (`LICENSE`).

---

## 1. What is and is not here

**The source corpus is not included.** It was retrieved from the Web of Science
Core Collection under a licence that forbids redistribution. Every table here is
*derived* from it, and every column that carried WoS record text — title,
abstract, author keywords, and the fused `text` field — has been removed:

| Source table | Column removed (WoS-proprietary) |
| --- | --- |
| `cleaned_data.csv` → `corpus/corpus_metadata.csv` | `TI`, `AB`, `DE`, `text` |
| `doc_topic_dist.csv` | `text` |
| `topics_info.csv` | `Representative_Docs` (verbatim abstracts) |
| `m6_validation_coded.csv`, `m6_intercoder_disagreements.csv` | `TI` |

What survives is copyright-free: publication year, country, journal, citation
count, DOI, and the model's own outputs (topic ids, probabilities, c-TF-IDF
scores).

Two ways to get back to the record level:

1. **Ask the authors.** The corpus itself cannot be posted here, but the
   corresponding author will supply it directly on reasonable request — see
   §6. This is the faster route and gives byte-identical input.
2. **Re-run the query.** `search_query.txt` carries the exact WoS query strings
   and retrieval parameters; run them against your own institutional
   subscription. Expect minor differences: WoS re-indexes continuously, so a
   later run will not reproduce the 2,332-record count exactly.

Mangrove area comes from the **HGMF 2020** map (Jia et al. 2023, 10 m), which is
publicly available — cite the source rather than redistributing the raster. Only
the per-country aggregate is deposited.

---

## 2. Directory map

```
data_pub/
├── search_query.txt          WoS queries (corpus + parent field) and retrieval parameters
├── corpus/
│   └── corpus_metadata.csv   record-level metadata, 2,332 publications
├── derived/
│   ├── topics/               topic model outputs, six-cluster taxonomy, evolution
│   ├── country_topic/        country x topic specialisation, LQ robustness, deficit
│   ├── country_area/         RRMI and its normalised variants
│   ├── temporal/             yearly counts, logistic fit
│   └── emerging/             candidate fronts, bootstrap CIs, sensitivity sweeps
├── robustness/               topic-quality and UMAP scans
├── validation/               human coding of study regions, annotator agreement
├── models/bertopic_model/    the frozen production BERTopic model
└── external/                 per-country mangrove area, parent-field country counts
```

---

## 3. Conventions that cause most mis-readings

**Three counts per country, three meanings.** They are never interchangeable:

| Name | Meaning | China | Indonesia |
| --- | --- | --- | --- |
| full count | publications credited once per country | 570 | 154 |
| fractional | weighted `1/(number of distinct countries)` | 468.2 | 108.3 |
| topic-assigned `N_c` | documents the topic model actually assigned | 325 | 67.7 |

**RRMI sign.** `RRMI = a_c − p_c` (area share minus publication share), so
**positive = area-rich / under-publishing**. The top-14 run from Indonesia
(+0.147) to Colombia (+0.014); China is the most negative (−0.222, rank 76 of 76).
"Low-indexed" therefore means an over-publishing method exporter, not an
under-studied country.

**Denominators.** The topic matrix covers **110** countries; the RRMI table has
**76**. Any statement spanning both must name both.

**2026 is partially indexed** (retrieval ended 15 July 2026, 160 records). It is
excluded from trend fitting and from the topic lifecycle throughout.

**Topic ids** (`Topic`, `T<id>`) refer to the frozen model in `models/`. They are
stable only against the deposited model; refitting on a different library
version renumbers them.

---

## 4. Table dictionary

### corpus/

| File | Rows | Columns | Notes |
| --- | --- | --- | --- |
| `corpus_metadata.csv` | 2,332 | `doc_id, PY, CU, SO, TC, DOI, Topic, prob_topic_0..4` | `doc_id` is the row index and joins to `doc_topic_dist.doc_index`. `CU` is `;`-separated; territories already folded (Taiwan→China, Puerto Rico→USA). DOI present for 2,164 records. `Topic = -1` marks the 771 unassigned documents. |

### derived/topics/

| File | Rows | Contents |
| --- | --- | --- |
| `topics_info.csv` | 73 | `Topic, Count, Name, Representation, LLM, Short_Label` — 72 topics plus the `-1` outlier row. `Representation` is c-TF-IDF keywords. |
| `topic_clusters.csv` | 72 | `Topic, Cluster, Cluster_Name, Docs, Share_2025, Share_2026_H1` — the frozen six-cluster Ward k=6 taxonomy; membership is re-derived and asserted on every run. |
| `cluster_summary.csv` | 6 | Per-cluster size, 2025 share, and the keyword strings used in Table 3. |
| `doc_topic_dist.csv` | 2,332 | `doc_index, Topic, prob_topic_0..4, PY` — document→topic assignment. |
| `topic_word_scores.csv` | 1,440 | `Topic, Rank, Word, c_TFIDF_Score` — top-20 words per topic. |
| `topic_coherence.csv` | 73 | `Topic, Count, Name, diversity, c_v_coherence`. |
| `topic_dynamics.csv` | 234 | `Topic, Words, Frequency, Timestamp, Proportion`. |
| `topic_evolution_proportions.csv` | 12 | Topic shares across the four phase-aligned windows `1985–1998 … 2023–2025` (Fig. 10). |
| `topic_network_edges.csv` | 188 | `source, target, similarity`. |
| `topic_evaluation.json` | — | Model-level diagnostics. |

### derived/country_topic/

| File | Rows | Contents |
| --- | --- | --- |
| `spatial_topic_si.csv` | 33 | Country × topic specialisation index (fractional counts). |
| `lq_robustness.csv` | 7 | `Country, Topic, Topic_Label, N_c, N_ck, observed_LQ, perm_p_value, shrunk_LQ, survives_permutation, survives_shrinkage` — the seven niches of Table 4, each with its permutation null and beta-binomial shrinkage. |
| `area_pub_deficit.csv` | 76 | `country, area, pubs, area_share, pub_share, deficit, rank`. Inner-joined on HGMF area, so zero-area method exporters never appear. |
| `study_area_vs_affiliation.csv` | 2,332 | `DI, PY, first_author_country, study_regions, n_regions, is_high_output, studies_low_output_tropical, has_region`. |
| `study_area_vs_affiliation_matrix.csv` | 175 | `first_author_country, study_region, N_pubs` cross-tab. |
| `study_area_fractional.csv` | 1,353 | Fractional study-region weights. |
| `multi_indicator_consistency.csv` | 76 | `Country, area, pubs, Citations, deficit, pub_density, citation_gap`. |
| `bivariate_attr_clean.csv` | 151 | `ISO3, country, mangrove_area_km2, n_pubs` — input to the 3×3 bivariate map. |

### derived/country_area/

| File | Rows | Contents |
| --- | --- | --- |
| `rrmi_parent_field.csv` | 76 | The headline RRMI (`rrmi`, `area_share`, `pub_share`) plus the parent-field normalization: `M_c, m_c, m_c_76, P_c, s_c, s_c_76, rrmi_norm, A, rrmi_norm_76, A_76`. `rrmi_norm = s_c − m_c`, `A = m_c/s_c`; **positive = over-attention** (the opposite convention to the headline RRMI). Reported on the world basis (primary) and re-normalized to the 76-country table (`*_76`). ΣP = 1,035,922 record counts. |
| `rrmi_normalization.csv` | 76 | Alternative denominators: all-field articles, GDP, population. R&D is coverage-only (57 of 76) and unusable. |
| `rrmi_window_2016_2025.csv` | 71 | RRMI recomputed on the 2016–2025 window. |

### derived/temporal/

| File | Rows | Contents |
| --- | --- | --- |
| `yearly_publications.csv` | 42 | `PY, annual, cumulative`, 1972–2026. |
| `logistic_growth_ci.json` | — | Complete years 1972–2025: `L = 13,145` (95% CI 9,126–17,163), `k = 0.14158 yr⁻¹`, `b = 288.32`, inflection `b/k = 2036.5` (CI 2033.7–2039.4), `R² = 0.9997`; phase boundaries 10% = 2021.0, 90% = 2052.0. **Do not quote** the deprecated "with truncated year" block (`K = 9,055 / t_mid = 2033.1`). |
| `logistic_inflection_profile.csv` | 64 | `inflection_year, best_L, RSS, delta_chi2` profile. |
| `logistic_growth_ci_report.md` | — | Fit diagnostics. |

### derived/emerging/

| File | Rows | Contents |
| --- | --- | --- |
| `emerging_tech_prototypes.csv` | 8 | The eight candidate fronts (T17, T16, T8, T4, T5, T57, T35, T68) with burst strength, interval, novelty/liveness flags. The older nine-prototype list (including T49) is superseded. **`W` is ordinal across the whole list, not within classes** — no candidate may be called "the strongest signal". |
| `prototype_bootstrap_ci.csv` | 8 | Bootstrap interval computed over the *re-detecting* replicates only, which is why it can exclude the null while T5/T35/T68 are re-detected in only ~41% of replicates. |
| `emerging_window_sensitivity.csv` | 216 | Window-configuration sweep. |
| `emerging_gamma_sensitivity.csv` | 216 | Kleinberg γ sweep. |
| `emerging_novelty_sensitivity.csv` | 216 | Novelty-window sweep. |
| `emerging_recency_sensitivity.csv` | 360 | Recency sweep. |

### robustness/

| File | Rows | Contents |
| --- | --- | --- |
| `topic_quality_robustness.csv` | 72 | `Topic, Count, Name, NPMI, npmi_rank_pct, is_bottom25, is_prototype`. |
| `topic_quality_robustness_umap_scan.csv` | 5 | `n_neighbors, n_topics, n_outliers, outlier_pct, n_docs`. |
| `vectorizer_comparison.json` | — | Evidence for pinning `CountVectorizer`. |

### validation/

| File | Rows | Contents |
| --- | --- | --- |
| `m6_validation_coded.csv` | 232 | Human coding of the study-region sample (title text removed). |
| `m6_three_coder_agreement.csv` | 6 | Three-coder agreement summary. |
| `m6_intercoder_disagreements.csv` | 10 | Disagreement cases. |
| `label_second_annotator.csv` | — | Second-annotator topic labels (Appendix A.2). |

### models/

`bertopic_model/` — the frozen production model (`config.json`,
`ctfidf.safetensors`, `ctfidf_config.json`, `topic_embeddings.safetensors`,
`topics.json`). Restore this rather than refitting: topic ids are stable only
against it.

### external/

| File | Rows | Contents |
| --- | --- | --- |
| `HGMF_country_area.csv` | 104 | Per-country mangrove area in km² (`sum` is the area column). |
| `parent_field_country_counts.xlsx` | 200 | WoS "Refine results → Countries/Regions" export for the parent-field query; input to the parent-field RRMI (Tables S7–S8). |

---

## 5. Reproducing from these tables

See `code_pub/README.md` §5 for the run order. The verification scripts in
`code_pub/scripts/` re-derive every number quoted in the manuscript from these
files and assert them; run them before re-quoting anything.

---

## 6. Availability and requests

**These tables: CC BY 4.0.** Free to use and redistribute with citation of the
paper and this deposit.

**The WoS source corpus: on request from the authors.** It was retrieved under a
Clarivate licence that forbids public redistribution, so it is deliberately not
in this archive. For verification, replication, or non-commercial academic
reuse, contact the corresponding author:

> Z.-W. Hou — hou_zhiwei@gmlab.ac.cn

Say briefly who you are, what you need it for, and whether you hold your own WoS
subscription (if you do, the query in `search_query.txt` will usually be
quicker). Requests are handled case by case; redistribution onward is not
permitted, since the licence is held by the subscriber, not by the authors.

Mangrove area is not ours to give either — it comes from the public **HGMF 2020**
map (Jia et al. 2023, 10 m); cite that source.
