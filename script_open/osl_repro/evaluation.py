"""Evaluation helpers for similarity search experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def zero_center_and_l2_normalize(vectors: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    centered = np.asarray(vectors, dtype=np.float32) - np.asarray(vectors, dtype=np.float32).mean(
        axis=0, keepdims=True
    )
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    return centered / np.maximum(norms, eps)


def topk_indices(similarity: np.ndarray, k: int) -> np.ndarray:
    if k >= similarity.shape[1]:
        return np.argsort(-similarity, axis=1)
    partition = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
    row_ids = np.arange(similarity.shape[0])[:, None]
    scores = similarity[row_ids, partition]
    order = np.argsort(-scores, axis=1)
    return partition[row_ids, order]


def precision_at_k_from_features(
    input_features: np.ndarray,
    output_features: np.ndarray,
    neighbor_k: int = 100,
) -> float:
    input_similarity = input_features @ input_features.T
    output_similarity = output_features @ output_features.T
    np.fill_diagonal(input_similarity, -np.inf)
    np.fill_diagonal(output_similarity, -np.inf)

    input_neighbors = topk_indices(input_similarity, neighbor_k)
    output_neighbors = topk_indices(output_similarity, neighbor_k)

    overlap = []
    for index in range(input_neighbors.shape[0]):
        gt = set(input_neighbors[index].tolist())
        pred = output_neighbors[index].tolist()
        overlap.append(sum(candidate in gt for candidate in pred) / float(neighbor_k))
    return float(np.mean(overlap))


@dataclass
class ExperimentMetrics:
    dataset: str
    num_train: int
    num_eval: int
    input_dim: int
    output_dim: int
    hash_length: int
    row_active: int
    neighbor_k: int
    osl_precision: float
    random_sparse_precision: float
    random_dense_precision: float
