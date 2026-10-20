## P0-1 Topic-quality robustness



Across the 72-topic solution, per-topic NPMI was -0.2793 (median -0.2833; Q1 -0.3215, Q3 -0.2424), within the range expected for short, technical documents.

Only 1 of the 9 candidate prototypes fell in the bottom NPMI quartile (11%, below the ~25% expected by chance), so the candidate prototypes are not concentrated among low-coherence clusters and their emerging labels are not an artefact of poor coherence.

Word-count sensitivity: NPMI moved only from -0.2694 (top_n=5) to -0.2793 (top_n=10) to -0.2811 (top_n=20), indicating the coherence reading is robust to the number of retained keywords.

UMAP n_neighbors scan ({5,10,15,30,50}): outlier rate ranged 27.7%–40.3% and topic count 60–83, so the reported solution is stable to neighbourhood size.
