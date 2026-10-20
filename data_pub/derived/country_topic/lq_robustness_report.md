# LQ Robustness (P0-3)

- Permutation null: 1000 replicates, one-sided (permuted LQ >= observed).
- Measurability threshold: N_c >= 10 topic-assigned fractional documents (unassigned documents excluded).
- Niche selection: top-7 positive RRMI among countries with N_c >= 10 topic-assigned fractional documents and >= 1000 km2 of mangrove.
- Countries re-derived from the current artifacts: Indonesia, Nigeria, Brazil, Mexico, Bangladesh, Colombia, Malaysia.
- Beta-binomial shrinkage: empirical-Bayes posterior LQ shrunk toward 1.

| Country | Topic | Label | N_c | N_ck | LQ_obs | p_perm | LQ_shrunk | survives |
|---|---|---|---|---|---|---|---|---|
| Colombia | T40 | Mangrove Remote Sensing Monitoring | 12.0 | 2.0 | 17.22 | 0.0050 | 8.95 | yes |
| Indonesia | T53 | Mangrove characterization and mapping | 67.7 | 8.2 | 15.70 | 0.0010 | 13.82 | yes |
| Malaysia | T18 | Mangrove Remote Sensing Applications | 32.5 | 7.0 | 13.32 | 0.0010 | 10.90 | yes |
| Bangladesh | T38 | Remote Sensing of Shoreline Change | 20.9 | 2.5 | 11.54 | 0.0030 | 5.08 | yes |
| Mexico | T70 | Mangrove leaf pigment spectroscopy | 47.4 | 3.5 | 11.41 | 0.0010 | 8.99 | yes |
| Nigeria | T50 | Tsunami impacts on coastal ecosystems | 10.5 | 1.0 | 11.33 | 0.0629 | 4.23 | no |
| Brazil | T15 | Amazon mud bank mangrove coasts | 68.8 | 12.1 | 9.69 | 0.0010 | 8.96 | yes |

All 7 niches survive the permutation null at p < 0.05: False.
All 7 niches retain shrunk LQ > 1: True (min shrunk LQ = 4.23).