#!/usr/bin/env python3
"""Run all methods on one processed dataset and report precision + timing."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_ROOT))

from osl_repro.datasets import list_processed_dataset_names, load_processed_dataset_arrays, processed_data_root  # noqa: E402
from osl_repro.evaluation import (  # noqa: E402
    neighbors_from_features,
    precision_at_k_from_neighbors,
    same_label_precision_at_k,
    zero_center_and_l2_normalize,
)
from osl_repro.model import (  # noqa: E402
    OSLConfig,
    OptimalSparseLifting,
    RandomDenseProjection,
    RandomSparseLifting,
)

DEFAULT_PLAN = {
    "num_train": 300,
    "num_eval": 1000,
    "neighbor_k": 100,
    "output_dim": 200,
    "hash_length": 8,
    "row_active": 6,
    "y_iters": 20,
    "w_iters": 20,
    "covariance_reg": 1e-2,
}

DATASET_OVERRIDES = {
    "mars_15d_band4": {
        "num_train": 100,
        "num_eval": 300,
        "neighbor_k": 50,
    }
}


def experiment_plan(dataset_name: str) -> dict:
    plan = dict(DEFAULT_PLAN)
    plan.update(DATASET_OVERRIDES.get(dataset_name, {}))
    return plan


def sample_train_eval(
    vectors,
    labels,
    num_train,
    num_eval,
    seed,
):
    vectors = zero_center_and_l2_normalize(vectors)
    total = num_train + num_eval
    rng = np.random.default_rng(seed)
    indices = rng.choice(vectors.shape[0], size=total, replace=False)
    train_ids = indices[:num_train]
    eval_ids = indices[num_train:]
    return vectors[train_ids], vectors[eval_ids], labels[eval_ids]


def timed_neighbors(features, neighbor_k):
    start = time.perf_counter()
    neighbors = neighbors_from_features(features, neighbor_k)
    return neighbors, time.perf_counter() - start


def metric_payload(output_neighbors, eval_labels, class_ids, input_neighbors, neighbor_k):
    return {
        "neighbor_k": neighbor_k,
        "precision_at_100": precision_at_k_from_neighbors(
            input_neighbors,
            output_neighbors,
            neighbor_k,
        ),
        "same_label_precision_at_100": same_label_precision_at_k(
            output_neighbors,
            eval_labels,
            class_ids,
        ),
    }


def run_osl_method(method_name, operator_metric, train_vectors, eval_vectors, eval_labels, class_ids, plan, seed, input_neighbors):
    config = OSLConfig(
        output_dim=plan["output_dim"],
        hash_length=plan["hash_length"],
        row_active=plan["row_active"],
        y_iters=plan["y_iters"],
        w_iters=plan["w_iters"],
        operator_metric=operator_metric,
        covariance_reg=plan["covariance_reg"],
        seed=seed,
    )
    start = time.perf_counter()
    model = OptimalSparseLifting(config).fit(train_vectors)
    fit_time = time.perf_counter() - start

    start = time.perf_counter()
    codes = model.encode(eval_vectors)
    encode_time = time.perf_counter() - start

    output_neighbors, query_time = timed_neighbors(codes, plan["neighbor_k"])
    result = {
        "method": method_name,
        "operator_metric": operator_metric,
        "fit_time_sec": fit_time,
        "encode_time_sec": encode_time,
        "query_time_sec": query_time,
        "fit_summary": model.export_fit_summary(),
    }
    result.update(
        metric_payload(
            output_neighbors=output_neighbors,
            eval_labels=eval_labels,
            class_ids=class_ids,
            input_neighbors=input_neighbors,
            neighbor_k=plan["neighbor_k"],
        )
    )
    return result


def run_random_sparse(train_vectors, eval_vectors, eval_labels, class_ids, plan, seed, input_neighbors):
    start = time.perf_counter()
    model = RandomSparseLifting(
        input_dim=train_vectors.shape[1],
        output_dim=plan["output_dim"],
        hash_length=plan["hash_length"],
        row_active=plan["row_active"],
        seed=seed,
    )
    fit_time = time.perf_counter() - start

    start = time.perf_counter()
    codes = model.encode(eval_vectors)
    encode_time = time.perf_counter() - start

    output_neighbors, query_time = timed_neighbors(codes, plan["neighbor_k"])
    result = {
        "method": "random_sparse",
        "operator_metric": None,
        "fit_time_sec": fit_time,
        "encode_time_sec": encode_time,
        "query_time_sec": query_time,
        "fit_summary": None,
    }
    result.update(
        metric_payload(
            output_neighbors=output_neighbors,
            eval_labels=eval_labels,
            class_ids=class_ids,
            input_neighbors=input_neighbors,
            neighbor_k=plan["neighbor_k"],
        )
    )
    return result


def run_random_dense(train_vectors, eval_vectors, eval_labels, class_ids, plan, seed, input_neighbors):
    start = time.perf_counter()
    model = RandomDenseProjection(
        input_dim=train_vectors.shape[1],
        output_dim=plan["output_dim"],
        hash_length=plan["hash_length"],
        seed=seed,
    )
    fit_time = time.perf_counter() - start

    start = time.perf_counter()
    codes = model.encode(eval_vectors)
    encode_time = time.perf_counter() - start

    output_neighbors, query_time = timed_neighbors(codes, plan["neighbor_k"])
    result = {
        "method": "random_dense",
        "operator_metric": None,
        "fit_time_sec": fit_time,
        "encode_time_sec": encode_time,
        "query_time_sec": query_time,
        "fit_summary": None,
    }
    result.update(
        metric_payload(
            output_neighbors=output_neighbors,
            eval_labels=eval_labels,
            class_ids=class_ids,
            input_neighbors=input_neighbors,
            neighbor_k=plan["neighbor_k"],
        )
    )
    return result

def adjust_plan(plan: dict, num_samples: int) -> dict:
    adjusted = dict(plan)
    if num_samples < 4:
        raise ValueError(f"dataset is too small for benchmarking: only {num_samples} samples")

    num_eval = min(adjusted["num_eval"], num_samples - 1)
    num_train = min(adjusted["num_train"], num_samples - num_eval)
    if num_train <= 0:
        num_train = max(1, num_samples // 3)
        num_eval = min(num_samples - num_train, num_eval)
    if num_eval <= 1:
        num_eval = num_samples - num_train
    if num_eval <= 1 or num_train <= 0:
        raise ValueError(
            f"could not build a valid split for {num_samples} samples "
            f"(num_train={num_train}, num_eval={num_eval})"
        )

    adjusted["num_train"] = num_train
    adjusted["num_eval"] = num_eval
    adjusted["neighbor_k"] = min(adjusted["neighbor_k"], num_eval - 1)
    return adjusted


def override_plan(plan: dict, args: argparse.Namespace) -> dict:
    updated = dict(plan)
    for key in [
        "num_train",
        "num_eval",
        "neighbor_k",
        "output_dim",
        "hash_length",
        "row_active",
        "y_iters",
        "w_iters",
        "covariance_reg",
    ]:
        value = getattr(args, key)
        if value is not None:
            updated[key] = value
    return updated


def run_benchmark(args: argparse.Namespace) -> dict:
    processed_root = args.data_dir.resolve()
    available = list_processed_dataset_names(processed_root)
    if args.dataset not in available:
        joined = ", ".join(sorted(available))
        raise KeyError(f"unknown dataset {args.dataset}. Available datasets: {joined}")

    bundle = load_processed_dataset_arrays(
        processed_root=processed_root,
        dataset_name=args.dataset,
        max_samples=args.max_samples,
    )
    vectors = bundle["vectors"]
    labels = bundle["labels"]
    class_ids = bundle["class_ids"]

    plan = override_plan(experiment_plan(args.dataset), args)
    plan = adjust_plan(plan, vectors.shape[0])

    train_vectors, eval_vectors, eval_labels = sample_train_eval(
        vectors=vectors,
        labels=labels,
        num_train=plan["num_train"],
        num_eval=plan["num_eval"],
        seed=args.seed,
    )
    input_neighbors, input_query_time = timed_neighbors(eval_vectors, plan["neighbor_k"])

    methods = [
        run_osl_method(
            "osl_euclidean",
            "euclidean",
            train_vectors,
            eval_vectors,
            eval_labels,
            class_ids,
            plan,
            args.seed,
            input_neighbors,
        ),
        run_osl_method(
            "osl_mahalanobis_diag",
            "mahalanobis_diag",
            train_vectors,
            eval_vectors,
            eval_labels,
            class_ids,
            plan,
            args.seed,
            input_neighbors,
        ),
        run_osl_method(
            "osl_mahalanobis_full",
            "mahalanobis_full",
            train_vectors,
            eval_vectors,
            eval_labels,
            class_ids,
            plan,
            args.seed,
            input_neighbors,
        ),
        run_random_sparse(
            train_vectors,
            eval_vectors,
            eval_labels,
            class_ids,
            plan,
            args.seed,
            input_neighbors,
        ),
        run_random_dense(
            train_vectors,
            eval_vectors,
            eval_labels,
            class_ids,
            plan,
            args.seed,
            input_neighbors,
        ),
    ]

    return {
        "dataset": args.dataset,
        "config": plan,
        "seed": args.seed,
        "num_samples": int(vectors.shape[0]),
        "input_dim": int(vectors.shape[1]),
        "num_labeled_eval": int((eval_labels > 0).sum()),
        "input_query_time_sec": input_query_time,
        "methods": methods,
    }


def print_summary(result: dict) -> None:
    print(f"dataset={result['dataset']} samples={result['num_samples']} dim={result['input_dim']}")
    print(
        f"train={result['config']['num_train']} eval={result['config']['num_eval']} "
        f"neighbor_k={result['config']['neighbor_k']}"
    )
    for row in result["methods"]:
        print(
            f"{row['method']:<22} "
            f"precision={row['precision_at_100']:.6f} "
            f"same_label={row['same_label_precision_at_100']} "
            f"fit={row['fit_time_sec']:.4f}s "
            f"encode={row['encode_time_sec']:.4f}s "
            f"query={row['query_time_sec']:.4f}s"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Processed dataset name in data/data_processed/manifest.json")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=processed_data_root(PROJECT_ROOT),
        help="Processed dataset directory.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-train", type=int, default=None)
    parser.add_argument("--num-eval", type=int, default=None)
    parser.add_argument("--neighbor-k", type=int, default=None)
    parser.add_argument("--output-dim", type=int, default=None)
    parser.add_argument("--hash-length", type=int, default=None)
    parser.add_argument("--row-active", type=int, default=None)
    parser.add_argument("--y-iters", type=int, default=None)
    parser.add_argument("--w-iters", type=int, default=None)
    parser.add_argument("--covariance-reg", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_benchmark(args)

    output_path = args.output
    if output_path is None:
        output_path = PROJECT_ROOT / "results" / f"{args.dataset}_benchmark.json"
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"[done] wrote {output_path}")
    print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
