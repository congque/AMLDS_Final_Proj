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


def neighbors_from_features(features: np.ndarray, neighbor_k: int = 100) -> np.ndarray:
    similarity = features @ features.T
    np.fill_diagonal(similarity, -np.inf)
    return topk_indices(similarity, neighbor_k)


def precision_at_k_from_neighbors(
    input_neighbors: np.ndarray,
    output_neighbors: np.ndarray,
    neighbor_k: int,
) -> float:
    overlap = []
    for index in range(input_neighbors.shape[0]):
        gt = set(input_neighbors[index].tolist())
        pred = output_neighbors[index].tolist()
        overlap.append(sum(candidate in gt for candidate in pred) / float(neighbor_k))
    return float(np.mean(overlap))


def precision_at_k_from_features(
    input_features: np.ndarray,
    output_features: np.ndarray,
    neighbor_k: int = 100,
) -> float:
    input_neighbors = neighbors_from_features(input_features, neighbor_k)
    output_neighbors = neighbors_from_features(output_features, neighbor_k)
    return precision_at_k_from_neighbors(input_neighbors, output_neighbors, neighbor_k)


def same_label_precision_at_k(
    output_neighbors: np.ndarray,
    labels: np.ndarray,
    class_ids: np.ndarray,
) -> float | None:
    if class_ids.size == 0:
        return None
    labels = np.asarray(labels)
    valid = np.isin(labels, class_ids)
    query_ids = np.flatnonzero(valid)
    if query_ids.size == 0:
        return None
    scores = []
    for index in query_ids:
        pred = output_neighbors[index]
        scores.append(float(np.mean(labels[pred] == labels[index])))
    return float(np.mean(scores))


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
