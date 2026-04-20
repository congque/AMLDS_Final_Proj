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
class IdxImageInfo:
    count: int
    rows: int
    cols: int


@dataclass
class IdxLabelInfo:
    count: int


@dataclass
class VecFileInfo:
    count: int
    dimension: int


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


@dataclass
class UnifiedDataset:
    dataset_name: str
    family_name: str
    unit_type: str
    vectors: np.ndarray
    labels: np.ndarray
    split_codes: np.ndarray
    split_names: np.ndarray
    database_indices: np.ndarray
    train_indices: np.ndarray
    query_indices: np.ndarray
    ground_truth_neighbors: np.ndarray
    coords: np.ndarray
    sample_names: np.ndarray
    sample_shape: np.ndarray
    class_ids: np.ndarray
    metadata_json: str

    @property
    def metadata(self) -> dict:
        return json.loads(self.metadata_json)


def read_idx_image_info(path: Path) -> IdxImageInfo:
    with path.open("rb") as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
    if magic != 2051:
        raise ValueError(f"unexpected IDX image magic number in {path}: {magic}")
    return IdxImageInfo(count=count, rows=rows, cols=cols)


def read_idx_label_info(path: Path) -> IdxLabelInfo:
    with path.open("rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
    if magic != 2049:
        raise ValueError(f"unexpected IDX label magic number in {path}: {magic}")
    return IdxLabelInfo(count=count)


def load_idx_images(path: Path) -> np.ndarray:
    info = read_idx_image_info(path)
    with path.open("rb") as handle:
        handle.seek(16)
        array = np.frombuffer(handle.read(), dtype=np.uint8)
    return array.reshape(info.count, info.rows, info.cols)


def load_idx_labels(path: Path) -> np.ndarray:
    info = read_idx_label_info(path)
    with path.open("rb") as handle:
        handle.seek(8)
        array = np.frombuffer(handle.read(), dtype=np.uint8)
    return array.reshape(info.count)


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


def read_vec_file_info(path: Path) -> VecFileInfo:
    with path.open("rb") as handle:
        dimension = struct.unpack("<i", handle.read(4))[0]
    if dimension <= 0:
        raise ValueError(f"invalid vector dimension in {path}: {dimension}")
    vector_bytes = 4 * (dimension + 1)
    file_size = path.stat().st_size
    if file_size % vector_bytes != 0:
        raise ValueError(
            f"file size {file_size} is not aligned with dimension {dimension} in {path}"
        )
    return VecFileInfo(count=file_size // vector_bytes, dimension=dimension)


def read_glove_info(path: Path) -> VecFileInfo:
    with path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline().strip()
    if not first_line:
        raise ValueError(f"empty GloVe file: {path}")
    dimension = len(first_line.split()) - 1
    if dimension <= 0:
        raise ValueError(f"failed to infer GloVe dimension from {path}")
    return VecFileInfo(count=-1, dimension=dimension)


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
    if vectors.ndim != 2:
        raise ValueError(f"expected vectors to be 2D, got shape {vectors.shape}")
    num_samples = vectors.shape[0]

    if labels is None:
        labels = np.full(num_samples, -1, dtype=np.int32)
    else:
        labels = np.asarray(labels, dtype=np.int32)
        if labels.shape != (num_samples,):
            raise ValueError(f"labels must have shape ({num_samples},), got {labels.shape}")

    if split_codes is None:
        split_codes = np.zeros(num_samples, dtype=np.int16)
    else:
        split_codes = np.asarray(split_codes, dtype=np.int16)
        if split_codes.shape != (num_samples,):
            raise ValueError(f"split_codes must have shape ({num_samples},), got {split_codes.shape}")

    if split_names is None:
        split_names = np.asarray(["all"])
    else:
        split_names = np.asarray(split_names)

    if database_indices is None:
        database_indices = np.arange(num_samples, dtype=np.int64)
    else:
        database_indices = np.asarray(database_indices, dtype=np.int64)

    if train_indices is None:
        train_indices = np.empty((0,), dtype=np.int64)
    else:
        train_indices = np.asarray(train_indices, dtype=np.int64)

    if query_indices is None:
        query_indices = np.empty((0,), dtype=np.int64)
    else:
        query_indices = np.asarray(query_indices, dtype=np.int64)

    if ground_truth_neighbors is None:
        ground_truth_neighbors = np.empty((0, 0), dtype=np.int64)
    else:
        ground_truth_neighbors = np.asarray(ground_truth_neighbors, dtype=np.int64)
        if ground_truth_neighbors.ndim != 2:
            raise ValueError(
                f"ground_truth_neighbors must be 2D, got shape {ground_truth_neighbors.shape}"
            )

    if coords is None:
        coords = np.empty((0, 2), dtype=np.int32)
    else:
        coords = np.asarray(coords, dtype=np.int32)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"coords must have shape (N, 2), got {coords.shape}")

    if sample_names is None:
        sample_names = np.empty((0,), dtype=np.str_)
    else:
        sample_names = np.asarray(sample_names)

    if sample_shape is None:
        sample_shape = np.empty((0,), dtype=np.int32)
    else:
        sample_shape = np.asarray(sample_shape, dtype=np.int32)

    if class_ids is None:
        class_ids = np.empty((0,), dtype=np.int32)
    else:
        class_ids = np.asarray(class_ids, dtype=np.int32)

    payload = {
        "schema_version": np.asarray(UNIFIED_DATASET_SCHEMA_VERSION, dtype=np.int32),
        "dataset_name": np.asarray(dataset_name),
        "family_name": np.asarray(family_name),
        "unit_type": np.asarray(unit_type),
        "feature_dim": np.asarray(vectors.shape[1], dtype=np.int32),
        "vectors": vectors,
        "labels": labels,
        "split_codes": split_codes,
        "split_names": split_names,
        "database_indices": database_indices,
        "train_indices": train_indices,
        "query_indices": query_indices,
        "ground_truth_neighbors": ground_truth_neighbors,
        "coords": coords,
        "sample_names": sample_names,
        "sample_shape": sample_shape,
        "class_ids": class_ids,
        "metadata_json": np.asarray(json.dumps(metadata or {}, sort_keys=True)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def load_unified_dataset(path: Path) -> UnifiedDataset:
    with np.load(path, allow_pickle=False) as data:
        return UnifiedDataset(
            dataset_name=str(data["dataset_name"].item()),
            family_name=str(data["family_name"].item()),
            unit_type=str(data["unit_type"].item()),
            vectors=np.asarray(data["vectors"], dtype=np.float32),
            labels=np.asarray(data["labels"], dtype=np.int32),
            split_codes=np.asarray(data["split_codes"], dtype=np.int16),
            split_names=np.asarray(data["split_names"]),
            database_indices=np.asarray(data["database_indices"], dtype=np.int64),
            train_indices=np.asarray(data["train_indices"], dtype=np.int64),
            query_indices=np.asarray(data["query_indices"], dtype=np.int64),
            ground_truth_neighbors=np.asarray(data["ground_truth_neighbors"], dtype=np.int64),
            coords=np.asarray(data["coords"], dtype=np.int32),
            sample_names=np.asarray(data["sample_names"]),
            sample_shape=np.asarray(data["sample_shape"], dtype=np.int32),
            class_ids=np.asarray(data["class_ids"], dtype=np.int32),
            metadata_json=str(data["metadata_json"].item()),
        )


def processed_data_root(project_root: Path) -> Path:
    return project_root / "data" / "data_processed"


def load_processed_manifest(processed_root: Path) -> dict:
    manifest_path = processed_root / "manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


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
    dataset = load_unified_dataset(processed_dataset_path(processed_root, dataset_name))
    vectors = dataset.vectors
    if max_samples is not None:
        vectors = vectors[:max_samples]
    return vectors
