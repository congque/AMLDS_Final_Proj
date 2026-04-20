# script_open

This folder is the shareable core of the current AMLDS final reproduction.

If someone only wants to understand the main experiment chain, the suggested
reading order is:

1. `run_basic_repro.py`
2. `osl_repro/model.py`
3. `osl_repro/evaluation.py`
4. `osl_repro/datasets.py`

## Paper-aligned pipeline

The code keeps the same high-level chain as the paper:

1. Load dataset vectors `X`
2. Solve an optimal sparse target code matrix `Y*`
3. Learn a sparse lifting operator `W*` so that `W* X ~= Y*`
4. Encode new samples with `W*`
5. Evaluate whether nearest neighbors are preserved

Where this appears in code:

- dataset loading: `run_basic_repro.py`
- paper step 1 (`Y*`): `osl_repro/model.py -> solve_optimal_sparse_codes`
- paper step 2 (`W*`): `osl_repro/model.py -> learn_sparse_lifting_operator`
- inference / encoding: `osl_repro/model.py -> encode_with_lifting_operator`
- retrieval metric: `osl_repro/evaluation.py -> precision_at_k_from_features`

## What is the same as the paper

- The training chain is still two-stage: first solve `Y*`, then solve `W*`.
- The target of step 1 is still to make `Y^T Y` approximate `X^T X`.
- The target of step 2 is still to make `W X` approximate `Y*`.
- New samples are still encoded by scoring with `W` and keeping top-`k` active outputs.
- The evaluation still uses neighbor overlap between the original space and the output space.

## What we changed on purpose

- We do **not** claim this is the official original implementation. This is our
  own reproduction organized to follow the paper's logic.
- We use a lightweight Frank-Wolfe-style solver with hard top-`k` projection.
- We do not implement the full regularized optimization exactly as written in
  the paper, such as the explicit `gamma` / `beta` sparsity terms.
- We also do not implement the full constrained solver for conditions like
  `0 <= W <= 1` and `W 1 = c 1` in an exact optimization sense. Instead, we
  enforce sparsity through the oracle updates and final hard top-`k` selection.
- Our default experiments are small-scale and intended to keep the pipeline
  easy to run and easy to inspect.
- We use unified `.npz` dataset files in `data/data_processed/` so that later
  experiments do not need to repeat raw preprocessing.

## File roles

- `run_basic_repro.py`
  Minimal experiment entry point. It loads one processed dataset, samples train
  and eval vectors, trains OSL, runs baselines, and writes metrics.

- `osl_repro/model.py`
  Core implementation. This is the best file to read if someone wants to match
  the paper's mathematical pipeline to the code.

- `osl_repro/evaluation.py`
  Similarity-search evaluation helpers. Right now the main metric is
  `precision@k` based on neighbor overlap.

- `osl_repro/datasets.py`
  Small helpers for reading the unified processed datasets and for exporting raw
  datasets into the common `.npz` schema.

- `export_unified_datasets.py`
  One-time preprocessing script. It converts all current raw datasets into one
  unified file per dataset.

## Short version

If someone asks "where is the main method?", the answer is:

- `osl_repro/model.py`

If someone asks "where is the whole experiment chain?", the answer is:

- `run_basic_repro.py`
