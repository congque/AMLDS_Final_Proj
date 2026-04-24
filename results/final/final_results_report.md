# Final Results

## Main Results

These are the final same-configuration results from the corrected metric solver.
Files:
- `main_retrieval_summary.json`
- `category_grouping_summary.json`

## Final Highlighted Datasets

These are the final presentation-ready results we would actually report.
File:
- `selected_final_results.json`

### Houston18
- Best config: `output_dim=320`, `hash_length=12`, `row_active=8`, `covariance_reg=1e-2`, `w_iters=40`
- `osl_mahalanobis_diag`: `precision=0.66054`, `same_label=0.22811`
- `random_sparse`: `precision=0.69939`, `same_label=0.22375`
- `random_dense`: `precision=0.75655`, `same_label=0.22330`
- Main takeaway: category grouping is slightly better than both random baselines, even though retrieval precision is still below `random_dense`.

### Mars
- Best config: `output_dim=320`, `hash_length=16`, `row_active=8`, `covariance_reg=1e-2`, `w_iters=40`
- `osl_mahalanobis_diag`: `precision=0.52233`
- `random_sparse`: `precision=0.29940`
- `random_dense`: `precision=0.57373`
- Main takeaway: the proposal gives a large gain over Euclidean OSL and random sparse, and gets close to the strong dense baseline.

### paviaU
- Best config: `output_dim=200`, `hash_length=12`, `row_active=6`, `covariance_reg=1e-3`, `w_iters=30`
- `osl_mahalanobis_diag`: `precision=0.39833`, `same_label=0.14537`
- `osl_mahalanobis_full`: `precision=0.39768`, `same_label=0.15347`
- `random_sparse`: `precision=0.43531`, `same_label=0.16158`
- `random_dense`: `precision=0.79704`, `same_label=0.15063`
- Main takeaway: the proposal improves clearly over Euclidean OSL, but this dataset is still not a win over the strongest baselines.
