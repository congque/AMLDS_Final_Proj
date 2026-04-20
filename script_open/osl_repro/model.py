"""Minimal but paper-aligned Optimal Sparse Lifting implementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


def _axis_binary_mask_from_topk(values: np.ndarray, count: int, axis: int) -> np.ndarray:
    if axis not in (0, 1):
        raise ValueError(f"unsupported axis: {axis}")
    if count <= 0:
        return np.zeros_like(values, dtype=np.float32)
    axis_size = values.shape[axis]
    if count >= axis_size:
        return np.ones_like(values, dtype=np.float32)

    indices = np.argpartition(values, kth=axis_size - count, axis=axis)[
        (slice(None),) * axis + (slice(axis_size - count, None),)
    ]
    mask = np.zeros_like(values, dtype=np.float32)
    if axis == 0:
        cols = np.arange(values.shape[1])[None, :]
        mask[indices, cols] = 1.0
    else:
        rows = np.arange(values.shape[0])[:, None]
        mask[rows, indices] = 1.0
    return mask


def _axis_binary_mask_from_bottomk(values: np.ndarray, count: int, axis: int) -> np.ndarray:
    return _axis_binary_mask_from_topk(-values, count=count, axis=axis)


def _column_topk_binary(scores: np.ndarray, count: int) -> np.ndarray:
    return _axis_binary_mask_from_topk(scores, count=count, axis=0)


def _row_topk_binary(scores: np.ndarray, count: int) -> np.ndarray:
    return _axis_binary_mask_from_topk(scores, count=count, axis=1)


def _column_linear_oracle(gradient: np.ndarray, count: int) -> np.ndarray:
    return _axis_binary_mask_from_bottomk(gradient, count=count, axis=0)


def _row_linear_oracle(gradient: np.ndarray, count: int) -> np.ndarray:
    return _axis_binary_mask_from_bottomk(gradient, count=count, axis=1)


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


def learn_sparse_lifting_operator(
    x: np.ndarray,
    target_codes: np.ndarray,
    output_dim: int,
    row_active: int,
    num_iters: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Paper step 2: learn a sparse lifting operator W* such that WX ~= Y*.

    Shapes:
    - x: (d, n)
    - target_codes: (m, n)
    - w: (m, d)

    The paper solves a sparse optimization for W. We keep the same two-stage
    pipeline and use a lightweight Frank-Wolfe-style row-sparse update.
    """

    input_dim = x.shape[0]
    w = np.zeros((output_dim, input_dim), dtype=np.float32)
    for row in range(output_dim):
        active = rng.choice(input_dim, size=row_active, replace=False)
        w[row, active] = 1.0

    for iteration in range(num_iters):
        # Same objective structure as the paper's second stage:
        # minimize ||W X - Y*||_F^2 over sparse W.
        residual = w @ x - target_codes
        gradient = residual @ x.T
        oracle = _row_linear_oracle(gradient, row_active)
        step = 2.0 / (iteration + 2.0)
        w = (1.0 - step) * w + step * oracle
    return _row_topk_binary(w, row_active)


def encode_with_lifting_operator(
    weight: np.ndarray,
    vectors: np.ndarray,
    hash_length: int,
) -> np.ndarray:
    """Paper inference step: score with W and keep the top-k active outputs."""

    x = np.asarray(vectors, dtype=np.float32).T
    scores = weight @ x
    return _column_topk_binary(scores, hash_length).T.astype(np.float32, copy=False)


@dataclass
class OSLConfig:
    output_dim: int = 256
    hash_length: int = 8
    row_active: int = 6
    y_iters: int = 40
    w_iters: int = 40
    seed: int = 0


class OptimalSparseLifting:
    """Two-stage OSL trainer aligned with the paper's solve-Y then solve-W chain."""

    def __init__(self, config: OSLConfig):
        self.config = config
        self.weight_: np.ndarray | None = None
        self.target_codes_: np.ndarray | None = None

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
        self.weight_ = learn_sparse_lifting_operator(
            x=x,
            target_codes=self.target_codes_,
            output_dim=self.config.output_dim,
            row_active=self.config.row_active,
            num_iters=self.config.w_iters,
            rng=rng,
        )
        return self

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        if self.weight_ is None:
            raise RuntimeError("model is not fitted yet")
        return encode_with_lifting_operator(
            weight=self.weight_,
            vectors=vectors,
            hash_length=self.config.hash_length,
        )

    def export_config(self) -> dict:
        return asdict(self.config)


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
        return _column_topk_binary(scores, self.hash_length).T.astype(np.float32, copy=False)


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
        return _column_topk_binary(scores, self.hash_length).T.astype(np.float32, copy=False)
