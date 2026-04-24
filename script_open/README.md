# script_open

This folder keeps the small set of files teammates need for dataset-level
benchmarking.

## Main entry point

Use:

```bash
python script_open/run_dataset_benchmark.py <dataset_name>
```

Examples:

```bash
python script_open/run_dataset_benchmark.py mnist
python script_open/run_dataset_benchmark.py cshsi_Houston18
python script_open/run_dataset_benchmark.py mars_15d_band4
```

## What it does

For one processed dataset, the script runs all methods:

- `osl_euclidean`
- `osl_mahalanobis_diag`
- `osl_mahalanobis_full`
- `random_sparse`
- `random_dense`

and reports:

- retrieval precision
- same-label precision when labels exist
- fit time
- encode time
- query time

## Files

- `run_dataset_benchmark.py`
  teammate-facing benchmark script
- `osl_repro/model.py`
  OSL and baseline implementations
- `osl_repro/evaluation.py`
  precision and same-label metrics
- `osl_repro/datasets.py`
  processed dataset loading

## Dataset requirement

The dataset name must exist in `data/data_processed/manifest.json`.
