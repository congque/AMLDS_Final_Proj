# AMLDS Final Project

This repo keeps the processed datasets plus the minimal code needed to run
Optimal Sparse Lifting benchmarks on one dataset at a time.

## What teammates need

- `data/data_processed/`
  one `.npz` file per dataset plus `manifest.json`
- `script_open/run_dataset_benchmark.py`
  one-command benchmark entry point
- `script_open/osl_repro/`
  model, evaluation, and dataset helpers

## Quick start

```bash
python script_open/run_dataset_benchmark.py mnist
python script_open/run_dataset_benchmark.py cshsi_Houston18
python script_open/run_dataset_benchmark.py mars_15d_band4
```

The script prints all methods and also writes a JSON file under `results/`.

## Final results

Clean final report-ready results are kept in:

- `results/final/final_results_report.md`
- `results/final/main_retrieval_summary.json`
- `results/final/category_grouping_summary.json`
- `results/final/selected_final_results.json`

## Output

For each method it reports:

- retrieval precision
- same-label precision when labels exist
- fit time
- encode time
- query time

## Supported methods

- `osl_euclidean`
- `osl_mahalanobis_diag`
- `osl_mahalanobis_full`
- `random_sparse`
- `random_dense`
