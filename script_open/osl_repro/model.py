"""Minimal, paper-aligned, but simplified Optimal Sparse Lifting implementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


def _column_topk_binary(scores: np.ndarray, count: int) -> np.ndarray:
    if count <= 0:
        return np.zeros_like(scores, dtype=np.float32)
    if count >= scores.shape[0]:
        return np.ones_like(scores, dtype=np.float32)
    indices = np.argpartition(scores, kth=scores.shape[0] - count, axis=0)[-count:]
    mask = np.zeros_like(scores, dtype=np.float32)
    cols = np.arange(scores.shape[1])[None, :]
    mask[indices, cols] = 1.0
    return mask


def _row_topk_binary(scores: np.ndarray, count: int) -> np.ndarray:
    if count <= 0:
        return np.zeros_like(scores, dtype=np.float32)
    if count >= scores.shape[1]:
        return np.ones_like(scores, dtype=np.float32)
    indices = np.argpartition(scores, kth=scores.shape[1] - count, axis=1)[:, -count:]
    mask = np.zeros_like(scores, dtype=np.float32)
    rows = np.arange(scores.shape[0])[:, None]
    mask[rows, indices] = 1.0
    return mask


def _column_linear_oracle(gradient: np.ndarray, count: int) -> np.ndarray:
    return _column_topk_binary(-gradient, count=count)


def _row_linear_oracle(gradient: np.ndarray, count: int) -> np.ndarray:
    return _row_topk_binary(-gradient, count=count)


def _global_topk_binary(scores: np.ndarray, count: int) -> np.ndarray:
    if count <= 0:
        return np.zeros_like(scores, dtype=np.float32)
    flat = scores.reshape(-1)
    if count >= flat.size:
        return np.ones_like(scores, dtype=np.float32)
    indices = np.argpartition(flat, kth=flat.size - count)[-count:]
    mask = np.zeros_like(flat, dtype=np.float32)
    mask[indices] = 1.0
    return mask.reshape(scores.shape)


def _hybrid_sparse_projection(weights: np.ndarray, total_active: int, min_per_row: int = 1) -> np.ndarray:
    if total_active <= 0:
        return np.zeros_like(weights, dtype=np.float32)

    output_dim, input_dim = weights.shape
    min_per_row = max(0, min(min_per_row, input_dim))
    min_budget = min(total_active, output_dim * min_per_row)
    base_mask = _row_topk_binary(np.abs(weights), min_per_row) if min_budget > 0 else np.zeros_like(
        weights,
        dtype=np.float32,
    )

    remaining = total_active - int(base_mask.sum())
    if remaining <= 0:
        return weights * base_mask

    free_scores = np.abs(weights) * (1.0 - base_mask)
    extra_mask = _global_topk_binary(free_scores, remaining)
    return weights * np.maximum(base_mask, extra_mask)


def _relative_frobenius_error(reference: np.ndarray, estimate: np.ndarray, eps: float = 1e-8) -> float:
    reference = np.asarray(reference, dtype=np.float32)
    estimate = np.asarray(estimate, dtype=np.float32)
    numerator = np.linalg.norm(reference - estimate)
    denominator = max(float(np.linalg.norm(reference)), eps)
    return float(numerator / denominator)


def solve_optimal_sparse_codes(
    x: np.ndarray,
    output_dim: int,
    hash_length: int,
    num_iters: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Paper step 1: solve a sparse target code matrix Y* for the training set.

    Shapes:
    - x: (d, n)
    - y: (m, n), where m is the lifted output dimension

    The paper optimizes Y so that Y^T Y approximates X^T X under sparsity
    constraints. Here we keep the same chain and use a lightweight
    Frank-Wolfe-style update with hard top-k binarization.
    """

    num_samples = x.shape[1]
    y = np.zeros((output_dim, num_samples), dtype=np.float32)
    for column in range(num_samples):
        active = rng.choice(output_dim, size=hash_length, replace=False)
        y[active, column] = 1.0

    gram = x.T @ x
    for iteration in range(num_iters):
        # Same objective structure as the paper's first stage:
        # minimize ||X^T X - Y^T Y||_F^2 over sparse Y.
        residual = y.T @ y - gram
        gradient = 2.0 * (y @ residual)
        oracle = _column_linear_oracle(gradient, hash_length)
        step = 2.0 / (iteration + 2.0)
        y = (1.0 - step) * y + step * oracle
    return _column_topk_binary(y, hash_length)


def estimate_mahalanobis_statistics(
    target_codes: np.ndarray,
    operator_metric: str,
    covariance_reg: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Estimate the covariance / precision used in the proposal's W-step.

    We keep the paper's first stage intact and estimate the covariance in the
    lifted code space from Y*. This makes the proposal change explicit:
    Euclidean W-step  ->  Mahalanobis-weighted W-step.
    """

    if operator_metric == "euclidean":
        return None, None

    centered_codes = target_codes - target_codes.mean(axis=1, keepdims=True)
    num_samples = max(1, centered_codes.shape[1] - 1)
    covariance = (centered_codes @ centered_codes.T) / float(num_samples)
    covariance = covariance.astype(np.float32, copy=False)
    identity = np.eye(covariance.shape[0], dtype=np.float32)
    covariance = covariance + np.float32(covariance_reg) * identity

    if operator_metric == "mahalanobis_diag":
        diagonal = np.maximum(np.diag(covariance), np.float32(covariance_reg))
        covariance = np.diag(diagonal).astype(np.float32, copy=False)
        precision = np.diag(1.0 / diagonal).astype(np.float32, copy=False)
        return covariance, precision

    if operator_metric == "mahalanobis_full":
        precision = np.linalg.pinv(covariance).astype(np.float32, copy=False)
        return covariance, precision

    raise ValueError(f"unsupported operator metric: {operator_metric}")


def learn_sparse_lifting_operator(
    x: np.ndarray,
    target_codes: np.ndarray,
    output_dim: int,
    row_active: int,
    num_iters: int,
    rng: np.random.Generator,
    operator_metric: str = "euclidean",
    covariance_reg: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Paper step 2: learn a sparse lifting operator W* such that WX ~= Y*.

    Shapes:
    - x: (d, n)
    - target_codes: (m, n)
    - w: (m, d)

    The paper solves a sparse optimization for W. We keep the same two-stage
    pipeline and use a lightweight Frank-Wolfe-style row-sparse update.

    Proposal extension:
    - original paper-aligned reproduction: Euclidean residual
    - our proposal: Mahalanobis-weighted residual in the lifted code space
    """

    input_dim = x.shape[0]
    covariance, precision = estimate_mahalanobis_statistics(
        target_codes=target_codes,
        operator_metric=operator_metric,
        covariance_reg=covariance_reg,
    )

    if operator_metric == "euclidean":
        w = np.zeros((output_dim, input_dim), dtype=np.float32)
        for row in range(output_dim):
            active = rng.choice(input_dim, size=row_active, replace=False)
            w[row, active] = 1.0

        for iteration in range(num_iters):
            residual = w @ x - target_codes
            gradient = residual @ x.T
            oracle = _row_linear_oracle(gradient, row_active)
            step = 2.0 / (iteration + 2.0)
            w = (1.0 - step) * w + step * oracle
        return _row_topk_binary(w, row_active), covariance, precision

    total_active = output_dim * row_active
    min_per_row = 1
    sample_count = max(1, x.shape[1])
    weighted_targets = target_codes if precision is None else precision @ target_codes
    w = (weighted_targets @ x.T) / float(sample_count)
    w = _hybrid_sparse_projection(
        w.astype(np.float32, copy=False),
        total_active=total_active,
        min_per_row=min_per_row,
    )

    x_scale = float(np.linalg.norm(x, ord="fro") ** 2 / float(sample_count))
    precision_scale = 1.0 if precision is None else float(np.linalg.norm(precision, ord=2))
    step_size = 1.0 / max(x_scale * precision_scale, 1e-6)

    for iteration in range(num_iters):
        residual = w @ x - target_codes
        weighted_residual = residual if precision is None else precision @ residual
        gradient = (weighted_residual @ x.T) / float(sample_count)
        candidate = w - step_size * gradient
        w = _hybrid_sparse_projection(candidate, total_active=total_active, min_per_row=min_per_row).astype(
            np.float32,
            copy=False,
        )
    return w, covariance, precision


def encode_with_lifting_operator(
    weight: np.ndarray,
    vectors: np.ndarray,
    hash_length: int,
) -> np.ndarray:
    """Paper inference step: score with W and keep the top-k active outputs."""

    x = np.asarray(vectors, dtype=np.float32).T
    scores = weight @ x
    return _column_topk_binary(scores, hash_length).T


@dataclass
class OSLConfig:
    output_dim: int = 256
    hash_length: int = 8
    row_active: int = 6
    y_iters: int = 40
    w_iters: int = 40
    operator_metric: str = "euclidean"
    covariance_reg: float = 1e-3
    seed: int = 0


class OptimalSparseLifting:
    """Two-stage OSL trainer following the paper's solve-Y then solve-W chain.

    The training order matches the paper, but the solver is a lightweight
    Frank-Wolfe-style approximation rather than the exact constrained optimizer.
    """

    def __init__(self, config: OSLConfig):
        self.config = config
        self.weight_: np.ndarray | None = None
        self.target_codes_: np.ndarray | None = None
        self.output_covariance_: np.ndarray | None = None
        self.output_precision_: np.ndarray | None = None
        self.fit_summary_: dict[str, float | str] = {}

    def fit(self, train_vectors: np.ndarray) -> "OptimalSparseLifting":
        x = np.asarray(train_vectors, dtype=np.float32).T
        rng = np.random.default_rng(self.config.seed)

        self.target_codes_ = solve_optimal_sparse_codes(
            x=x,
            output_dim=self.config.output_dim,
            hash_length=self.config.hash_length,
            num_iters=self.config.y_iters,
            rng=rng,
        )
        self.weight_, self.output_covariance_, self.output_precision_ = learn_sparse_lifting_operator(
            x=x,
            target_codes=self.target_codes_,
            output_dim=self.config.output_dim,
            row_active=self.config.row_active,
            num_iters=self.config.w_iters,
            rng=rng,
            operator_metric=self.config.operator_metric,
            covariance_reg=self.config.covariance_reg,
        )
        self.fit_summary_ = self._build_fit_summary(x)
        return self

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        if self.weight_ is None:
            raise RuntimeError("model is not fitted yet")
        return encode_with_lifting_operator(
            weight=self.weight_,
            vectors=vectors,
            hash_length=self.config.hash_length,
        )

    def _build_fit_summary(self, x: np.ndarray) -> dict[str, float | str]:
        if self.target_codes_ is None or self.weight_ is None:
            raise RuntimeError("fit summary is only available after fitting")

        y_similarity_error = _relative_frobenius_error(x.T @ x, self.target_codes_.T @ self.target_codes_)
        residual = self.weight_ @ x - self.target_codes_
        euclidean_error = float(np.linalg.norm(residual) / max(1, residual.shape[1]))
        weighted_error = euclidean_error
        covariance_trace = 0.0
        precision_trace = 0.0
        if self.output_precision_ is not None:
            weighted_error = float(
                np.sum(residual * (self.output_precision_ @ residual)) / max(1, residual.shape[1])
            )
        if self.output_covariance_ is not None:
            covariance_trace = float(np.trace(self.output_covariance_))
        if self.output_precision_ is not None:
            precision_trace = float(np.trace(self.output_precision_))
        return {
            "operator_metric": self.config.operator_metric,
            "y_similarity_error": y_similarity_error,
            "w_euclidean_error": euclidean_error,
            "w_weighted_error": weighted_error,
            "covariance_reg": float(self.config.covariance_reg),
            "covariance_trace": covariance_trace,
            "precision_trace": precision_trace,
        }

    def export_config(self) -> dict:
        return asdict(self.config)

    def export_fit_summary(self) -> dict[str, float | str]:
        return dict(self.fit_summary_)


class RandomSparseLifting:
    """Fly-style random sparse lifting baseline."""

    def __init__(self, input_dim: int, output_dim: int, hash_length: int, row_active: int, seed: int = 0):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hash_length = hash_length
        self.row_active = row_active
        self.seed = seed
        self.weight_ = self._build_weight()

    def _build_weight(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        weight = np.zeros((self.output_dim, self.input_dim), dtype=np.float32)
        for row in range(self.output_dim):
            active = rng.choice(self.input_dim, size=self.row_active, replace=False)
            weight[row, active] = 1.0
        return weight

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        x = np.asarray(vectors, dtype=np.float32).T
        scores = self.weight_ @ x
        return _column_topk_binary(scores, self.hash_length).T


class RandomDenseProjection:
    """Dense random projection baseline with top-k binarization."""

    def __init__(self, input_dim: int, output_dim: int, hash_length: int, seed: int = 0):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hash_length = hash_length
        rng = np.random.default_rng(seed)
        self.weight_ = rng.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(max(1, input_dim)),
            size=(output_dim, input_dim),
        ).astype(np.float32)

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        x = np.asarray(vectors, dtype=np.float32).T
        scores = self.weight_ @ x
        return _column_topk_binary(scores, self.hash_length).T
