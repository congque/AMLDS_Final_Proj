#!/usr/bin/env python3
"""Run the basic OSL reproduction from unified processed datasets.

Paper-aligned chain:
1. load processed vectors X
2. sample a train set and an evaluation set
3. solve Y* and then learn W* on the train set
4. encode evaluation samples with W*
5. compare neighbor overlap in the original space and output space
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from osl_repro.datasets import (  # noqa: E402
    load_processed_vectors,
    processed_data_root,
)
from osl_repro.evaluation import (  # noqa: E402
    ExperimentMetrics,
    precision_at_k_from_features,
    zero_center_and_l2_normalize,
)
from osl_repro.model import (  # noqa: E402
    OSLConfig,
    OptimalSparseLifting,
    RandomDenseProjection,
    RandomSparseLifting,
)


DATASET_ALIASES = {
    "mnist": "mnist",
    "glove": "glove_6b_300d",
    "glove_6b_300d": "glove_6b_300d",
    "sift": "sift1m",
    "sift1m": "sift1m",
    "mars": "mars_15d_band4",
    "mars_15d_band4": "mars_15d_band4",
    "Houston13": "cshsi_Houston13",
    "Houston18": "cshsi_Houston18",
    "paviaC": "cshsi_paviaC",
    "paviaU": "cshsi_paviaU",
    "Dioni": "cshsi_Dioni",
    "Loukia": "cshsi_Loukia",
    "cshsi_Houston13": "cshsi_Houston13",
    "cshsi_Houston18": "cshsi_Houston18",
    "cshsi_paviaC": "cshsi_paviaC",
    "cshsi_paviaU": "cshsi_paviaU",
    "cshsi_Dioni": "cshsi_Dioni",
    "cshsi_Loukia": "cshsi_Loukia",
}


def resolve_dataset_name(name: str) -> str:
    return DATASET_ALIASES[name]


def sample_train_eval(
    vectors: np.ndarray,
    num_train: int,
    num_eval: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    vectors = zero_center_and_l2_normalize(vectors)
    total_needed = num_train + num_eval
    if total_needed > vectors.shape[0]:
        raise ValueError(
            f"requested {total_needed} samples but dataset only provides {vectors.shape[0]}"
        )
    rng = np.random.default_rng(seed)
    indices = rng.choice(vectors.shape[0], size=total_needed, replace=False)
    train_ids = indices[:num_train]
    eval_ids = indices[num_train:]
    return vectors[train_ids], vectors[eval_ids]


def evaluate_codes(eval_vectors: np.ndarray, codes: np.ndarray, neighbor_k: int) -> float:
    return precision_at_k_from_features(
        input_features=eval_vectors,
        output_features=codes,
        neighbor_k=neighbor_k,
    )


def run_experiment(args: argparse.Namespace) -> dict:
    dataset_name = resolve_dataset_name(args.dataset)
    processed_root = args.data_dir.resolve()
    print(f"[info] step 1/5 loading dataset={dataset_name}")
    raw_vectors = load_processed_vectors(
        processed_root=processed_root,
        dataset_name=dataset_name,
        max_samples=args.max_samples,
    )
    print(f"[info] raw shape={raw_vectors.shape}")

    print("[info] step 2/5 normalizing and sampling train/eval splits")
    train_vectors, eval_vectors = sample_train_eval(
        vectors=raw_vectors,
        num_train=args.num_train,
        num_eval=args.num_eval,
        seed=args.seed,
    )
    print(f"[info] train shape={train_vectors.shape} eval shape={eval_vectors.shape}")

    config = OSLConfig(
        output_dim=args.output_dim,
        hash_length=args.hash_length,
        row_active=args.row_active,
        y_iters=args.y_iters,
        w_iters=args.w_iters,
        seed=args.seed,
    )

    print("[info] step 3/5 fitting OSL (solve Y* then learn W*)")
    osl = OptimalSparseLifting(config).fit(train_vectors)
    print("[info] step 4/5 encoding evaluation vectors")
    osl_codes = osl.encode(eval_vectors)

    print("[info] building random sparse baseline")
    random_sparse = RandomSparseLifting(
        input_dim=eval_vectors.shape[1],
        output_dim=args.output_dim,
        hash_length=args.hash_length,
        row_active=args.row_active,
        seed=args.seed,
    )
    random_sparse_codes = random_sparse.encode(eval_vectors)

    print("[info] building random dense baseline")
    random_dense = RandomDenseProjection(
        input_dim=eval_vectors.shape[1],
        output_dim=args.output_dim,
        hash_length=args.hash_length,
        seed=args.seed,
    )
    random_dense_codes = random_dense.encode(eval_vectors)

    print("[info] step 5/5 evaluating precision@k")
    metrics = ExperimentMetrics(
        dataset=dataset_name,
        num_train=args.num_train,
        num_eval=args.num_eval,
        input_dim=eval_vectors.shape[1],
        output_dim=args.output_dim,
        hash_length=args.hash_length,
        row_active=args.row_active,
        neighbor_k=args.neighbor_k,
        osl_precision=evaluate_codes(eval_vectors, osl_codes, args.neighbor_k),
        random_sparse_precision=evaluate_codes(eval_vectors, random_sparse_codes, args.neighbor_k),
        random_dense_precision=evaluate_codes(eval_vectors, random_dense_codes, args.neighbor_k),
    )

    serializable_args = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    return {
        "config": serializable_args,
        "canonical_dataset": dataset_name,
        "osl_config": osl.export_config(),
        "metrics": asdict(metrics),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASET_ALIASES), default="mnist")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=processed_data_root(PROJECT_ROOT),
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-train", type=int, default=500)
    parser.add_argument("--num-eval", type=int, default=1000)
    parser.add_argument("--output-dim", type=int, default=256)
    parser.add_argument("--hash-length", type=int, default=8)
    parser.add_argument("--row-active", type=int, default=6)
    parser.add_argument("--neighbor-k", type=int, default=100)
    parser.add_argument("--y-iters", type=int, default=40)
    parser.add_argument("--w-iters", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "basic_repro.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_experiment(args)
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[done] wrote results to {output_path}")
    print(json.dumps(result["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
