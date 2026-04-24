"""Dataset readers and metadata helpers for the OSL reproduction."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Optional

import h5py
import numpy as np
from scipy.io import loadmat
import tifffile


@dataclass
class CshsiSceneSpec:
    scene_name: str
    data_path: Path
    gt_path: Path


@dataclass
class CshsiSceneData:
    scene_name: str
    cube: np.ndarray
    labels: np.ndarray


UNIFIED_DATASET_SCHEMA_VERSION = 1


def load_idx_images(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise ValueError(f"unexpected IDX image magic number in {path}: {magic}")
        array = np.frombuffer(handle.read(), dtype=np.uint8)
    return array.reshape(count, rows, cols)


def load_idx_labels(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError(f"unexpected IDX label magic number in {path}: {magic}")
        array = np.frombuffer(handle.read(), dtype=np.uint8)
    return array.reshape(count)


def mnist_root(data_root: Path) -> Path:
    return data_root / "mnist"


def glove_path(data_root: Path) -> Path:
    return data_root / "glove_6b" / "glove.6B.300d.txt"


def sift_root(data_root: Path) -> Path:
    return data_root / "sift1m" / "sift"


def read_fvecs(path: Path, max_vectors: Optional[int] = None) -> np.ndarray:
    with path.open("rb") as handle:
        dimension = struct.unpack("<i", handle.read(4))[0]
    raw = np.fromfile(path, dtype=np.int32 if dimension < 0 else np.float32)
    if raw.size == 0:
        return np.empty((0, 0), dtype=np.float32)
    dimension = raw.view(np.int32)[0]
    stride = dimension + 1
    vectors = raw.reshape(-1, stride)[:, 1:].astype(np.float32, copy=False)
    if max_vectors is not None:
        vectors = vectors[:max_vectors]
    return vectors


def read_ivecs(path: Path, max_vectors: Optional[int] = None) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        return np.empty((0, 0), dtype=np.int32)
    dimension = int(raw[0])
    stride = dimension + 1
    vectors = raw.reshape(-1, stride)[:, 1:]
    if max_vectors is not None:
        vectors = vectors[:max_vectors]
    return vectors
def load_glove_entries(path: Path, max_vectors: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
    tokens = []
    vectors = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if max_vectors is not None and len(vectors) >= max_vectors:
                break
            stripped = line.strip()
            if not stripped:
                continue
            split_at = stripped.find(" ")
            if split_at <= 0:
                raise ValueError(f"malformed GloVe line {line_number} in {path}")
            token = stripped[:split_at]
            values = stripped[split_at + 1 :]
            vector = np.fromstring(values, sep=" ", dtype=np.float32)
            if vector.size == 0:
                raise ValueError(f"empty vector at line {line_number} in {path}")
            tokens.append(token)
            vectors.append(vector)
    if not vectors:
        raise ValueError(f"no vectors loaded from {path}")
    return np.asarray(tokens), np.vstack(vectors)


def cshsi_root(data_root: Path) -> Path:
    return data_root / "cshsi" / "datasets"


def list_cshsi_scenes(data_root: Path) -> dict[str, CshsiSceneSpec]:
    root = cshsi_root(data_root)
    return {
        "Houston13": CshsiSceneSpec(
            scene_name="Houston13",
            data_path=root / "Houston" / "Houston13.mat",
            gt_path=root / "Houston" / "Houston13_7gt.mat",
        ),
        "Houston18": CshsiSceneSpec(
            scene_name="Houston18",
            data_path=root / "Houston" / "Houston18.mat",
            gt_path=root / "Houston" / "Houston18_7gt.mat",
        ),
        "paviaC": CshsiSceneSpec(
            scene_name="paviaC",
            data_path=root / "Pavia" / "paviaC.mat",
            gt_path=root / "Pavia" / "paviaC_7gt.mat",
        ),
        "paviaU": CshsiSceneSpec(
            scene_name="paviaU",
            data_path=root / "Pavia" / "paviaU.mat",
            gt_path=root / "Pavia" / "paviaU_7gt.mat",
        ),
        "Dioni": CshsiSceneSpec(
            scene_name="Dioni",
            data_path=root / "HyRANK" / "Dioni.mat",
            gt_path=root / "HyRANK" / "Dioni_gt_out68.mat",
        ),
        "Loukia": CshsiSceneSpec(
            scene_name="Loukia",
            data_path=root / "HyRANK" / "Loukia.mat",
            gt_path=root / "HyRANK" / "Loukia_gt_out68.mat",
        ),
    }


def _load_mat_array(path: Path, key: str) -> np.ndarray:
    try:
        obj = loadmat(path)
        if key not in obj:
            raise KeyError(f"key {key} not found in {path}")
        return np.asarray(obj[key])
    except NotImplementedError:
        # Houston data is stored as MATLAB v7.3 / HDF5, so scipy falls back here.
        with h5py.File(path, "r") as handle:
            if key not in handle:
                raise KeyError(f"key {key} not found in {path}")
            array = np.asarray(handle[key])
        axes = tuple(range(array.ndim - 1, -1, -1))
        return np.transpose(array, axes=axes)


def load_cshsi_scene(data_root: Path, scene_name: str) -> CshsiSceneData:
    scene_map = list_cshsi_scenes(data_root)
    if scene_name not in scene_map:
        available = ", ".join(sorted(scene_map))
        raise KeyError(f"unknown CSHSI scene {scene_name}; available scenes: {available}")
    spec = scene_map[scene_name]
    cube = _load_mat_array(spec.data_path, "ori_data").astype(np.float32, copy=False)
    labels = _load_mat_array(spec.gt_path, "map").astype(np.int16, copy=False)
    if cube.ndim != 3:
        raise ValueError(f"expected a 3D cube in {spec.data_path}, got shape {cube.shape}")
    if labels.ndim != 2:
        raise ValueError(f"expected a 2D label map in {spec.gt_path}, got shape {labels.shape}")
    if cube.shape[:2] != labels.shape:
        raise ValueError(
            f"spatial shape mismatch for {scene_name}: cube {cube.shape[:2]} vs labels {labels.shape}"
        )
    return CshsiSceneData(scene_name=scene_name, cube=cube, labels=labels)


def flatten_hsi_cube(
    cube: np.ndarray,
    labels: np.ndarray,
    labeled_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width, bands = cube.shape
    vectors = cube.reshape(height * width, bands).astype(np.float32, copy=False)
    flat_labels = labels.reshape(height * width).astype(np.int16, copy=False)
    rows, cols = np.indices((height, width), dtype=np.int32)
    coords = np.stack([rows.reshape(-1), cols.reshape(-1)], axis=1)
    if labeled_only:
        keep = flat_labels != 0
        vectors = vectors[keep]
        flat_labels = flat_labels[keep]
        coords = coords[keep]
    return vectors, flat_labels, coords


def load_mars_band_vectors(
    image_dir: Path,
    band_index: int = 3,
    max_images: Optional[int] = None,
) -> tuple[np.ndarray, list[str]]:
    paths = sorted(image_dir.glob("*.tif"))
    if not paths:
        raise FileNotFoundError(f"no TIFF files found in {image_dir}")
    if max_images is not None:
        paths = paths[:max_images]

    vectors = []
    names = []
    for path in paths:
        image = tifffile.imread(path)
        if image.ndim != 3:
            raise ValueError(f"expected 3D TIFF in {path}, got shape {image.shape}")
        if not (0 <= band_index < image.shape[2]):
            raise IndexError(
                f"band index {band_index} is out of range for {path} with shape {image.shape}"
            )
        band = image[:, :, band_index].astype(np.float32, copy=False)
        vectors.append(band.reshape(-1))
        names.append(path.name)
    return np.vstack(vectors), names


def mars_root(data_root: Path) -> Path:
    return data_root / "mars" / "images_15d"


def parse_mars_tile_name(name: str) -> tuple[int, int]:
    match = re.fullmatch(r"col(\d+)_row(\d+)\.tif", name)
    if match is None:
        raise ValueError(f"unexpected Mars tile name: {name}")
    col = int(match.group(1))
    row = int(match.group(2))
    return row, col


def save_unified_dataset(
    path: Path,
    *,
    dataset_name: str,
    family_name: str,
    unit_type: str,
    vectors: np.ndarray,
    labels: Optional[np.ndarray] = None,
    split_codes: Optional[np.ndarray] = None,
    split_names: Optional[np.ndarray] = None,
    database_indices: Optional[np.ndarray] = None,
    train_indices: Optional[np.ndarray] = None,
    query_indices: Optional[np.ndarray] = None,
    ground_truth_neighbors: Optional[np.ndarray] = None,
    coords: Optional[np.ndarray] = None,
    sample_names: Optional[np.ndarray] = None,
    sample_shape: Optional[np.ndarray] = None,
    class_ids: Optional[np.ndarray] = None,
    metadata: Optional[dict] = None,
) -> None:
    vectors = np.asarray(vectors, dtype=np.float32)
    num_samples = vectors.shape[0]

    payload = {
        "schema_version": np.asarray(UNIFIED_DATASET_SCHEMA_VERSION, dtype=np.int32),
        "dataset_name": np.asarray(dataset_name),
        "family_name": np.asarray(family_name),
        "unit_type": np.asarray(unit_type),
        "feature_dim": np.asarray(vectors.shape[1], dtype=np.int32),
        "vectors": vectors,
        "labels": np.full(num_samples, -1, dtype=np.int32) if labels is None else np.asarray(labels, dtype=np.int32),
        "split_codes": np.zeros(num_samples, dtype=np.int16)
        if split_codes is None
        else np.asarray(split_codes, dtype=np.int16),
        "split_names": np.asarray(["all"]) if split_names is None else np.asarray(split_names),
        "database_indices": np.arange(num_samples, dtype=np.int64)
        if database_indices is None
        else np.asarray(database_indices, dtype=np.int64),
        "train_indices": np.empty((0,), dtype=np.int64)
        if train_indices is None
        else np.asarray(train_indices, dtype=np.int64),
        "query_indices": np.empty((0,), dtype=np.int64)
        if query_indices is None
        else np.asarray(query_indices, dtype=np.int64),
        "ground_truth_neighbors": np.empty((0, 0), dtype=np.int64)
        if ground_truth_neighbors is None
        else np.asarray(ground_truth_neighbors, dtype=np.int64),
        "coords": np.empty((0, 2), dtype=np.int32) if coords is None else np.asarray(coords, dtype=np.int32),
        "sample_names": np.empty((0,), dtype=np.str_) if sample_names is None else np.asarray(sample_names),
        "sample_shape": np.empty((0,), dtype=np.int32)
        if sample_shape is None
        else np.asarray(sample_shape, dtype=np.int32),
        "class_ids": np.empty((0,), dtype=np.int32) if class_ids is None else np.asarray(class_ids, dtype=np.int32),
        "metadata_json": np.asarray(json.dumps(metadata or {}, sort_keys=True)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def processed_data_root(project_root: Path) -> Path:
    return project_root / "data" / "data_processed"


def load_processed_manifest(processed_root: Path) -> dict:
    manifest_path = processed_root / "manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def list_processed_dataset_names(processed_root: Path) -> list[str]:
    manifest = load_processed_manifest(processed_root)
    return [item["dataset_name"] for item in manifest["datasets"]]


def processed_dataset_path(processed_root: Path, dataset_name: str) -> Path:
    manifest = load_processed_manifest(processed_root)
    for item in manifest["datasets"]:
        if item["dataset_name"] == dataset_name:
            return Path(item["output_path"])
    raise KeyError(f"unknown processed dataset: {dataset_name}")


def load_processed_vectors(
    processed_root: Path,
    dataset_name: str,
    max_samples: Optional[int] = None,
) -> np.ndarray:
    with np.load(processed_dataset_path(processed_root, dataset_name), allow_pickle=False) as data:
        vectors = np.asarray(data["vectors"], dtype=np.float32)
    if max_samples is not None:
        vectors = vectors[:max_samples]
    return vectors


def load_processed_dataset_arrays(
    processed_root: Path,
    dataset_name: str,
    max_samples: Optional[int] = None,
) -> dict[str, np.ndarray]:
    with np.load(processed_dataset_path(processed_root, dataset_name), allow_pickle=False) as data:
        vectors = np.asarray(data["vectors"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=np.int32)
        class_ids = np.asarray(data["class_ids"], dtype=np.int32)
    if max_samples is not None:
        vectors = vectors[:max_samples]
        labels = labels[:max_samples]
    return {
        "vectors": vectors,
        "labels": labels,
        "class_ids": class_ids,
    }
