#!/usr/bin/env python3
"""Export all AMLDS_final datasets into a unified single-file format."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from osl_repro.datasets import (  # noqa: E402
    flatten_hsi_cube,
    glove_path,
    load_cshsi_scene,
    load_glove_entries,
    load_idx_images,
    load_idx_labels,
    load_mars_band_vectors,
    list_cshsi_scenes,
    mars_root,
    mnist_root,
    parse_mars_tile_name,
    read_fvecs,
    read_ivecs,
    save_unified_dataset,
    sift_root,
)


def export_mnist(raw_root: Path, output_dir: Path) -> dict:
    root = mnist_root(raw_root)
    train_images = load_idx_images(root / "train-images-idx3-ubyte").reshape(60000, -1)
    test_images = load_idx_images(root / "t10k-images-idx3-ubyte").reshape(10000, -1)
    train_labels = load_idx_labels(root / "train-labels-idx1-ubyte")
    test_labels = load_idx_labels(root / "t10k-labels-idx1-ubyte")

    vectors = np.concatenate([train_images, test_images], axis=0).astype(np.float32) / 255.0
    labels = np.concatenate([train_labels, test_labels], axis=0).astype(np.int32)
    split_codes = np.concatenate(
        [
            np.zeros(train_images.shape[0], dtype=np.int16),
            np.ones(test_images.shape[0], dtype=np.int16),
        ]
    )
    split_names = np.asarray(["train", "test"])
    train_indices = np.arange(train_images.shape[0], dtype=np.int64)
    query_indices = np.arange(train_images.shape[0], vectors.shape[0], dtype=np.int64)
    output_path = output_dir / "mnist.npz"
    save_unified_dataset(
        output_path,
        dataset_name="mnist",
        family_name="mnist",
        unit_type="image",
        vectors=vectors,
        labels=labels,
        split_codes=split_codes,
        split_names=split_names,
        database_indices=train_indices,
        train_indices=train_indices,
        query_indices=query_indices,
        sample_shape=np.asarray([28, 28], dtype=np.int32),
        class_ids=np.arange(10, dtype=np.int32),
        metadata={
            "raw_root": str(root),
            "input_normalization": "divide_by_255",
            "official_split_sizes": {"train": 60000, "test": 10000},
        },
    )
    return {
        "dataset_name": "mnist",
        "output_path": str(output_path),
        "num_samples": int(vectors.shape[0]),
        "feature_dim": int(vectors.shape[1]),
        "database_size": int(train_indices.size),
        "query_size": int(query_indices.size),
    }


def export_glove(raw_root: Path, output_dir: Path) -> dict:
    path = glove_path(raw_root)
    tokens, vectors = load_glove_entries(path)
    output_path = output_dir / "glove_6b_300d.npz"
    save_unified_dataset(
        output_path,
        dataset_name="glove_6b_300d",
        family_name="glove",
        unit_type="embedding",
        vectors=vectors,
        split_codes=np.zeros(vectors.shape[0], dtype=np.int16),
        split_names=np.asarray(["corpus"]),
        database_indices=np.arange(vectors.shape[0], dtype=np.int64),
        sample_names=tokens,
        metadata={
            "source_file": str(path),
            "embedding_dim": int(vectors.shape[1]),
            "num_tokens": int(vectors.shape[0]),
        },
    )
    return {
        "dataset_name": "glove_6b_300d",
        "output_path": str(output_path),
        "num_samples": int(vectors.shape[0]),
        "feature_dim": int(vectors.shape[1]),
        "database_size": int(vectors.shape[0]),
        "query_size": 0,
    }


def export_sift1m(raw_root: Path, output_dir: Path) -> dict:
    root = sift_root(raw_root)
    base = read_fvecs(root / "sift_base.fvecs")
    learn = read_fvecs(root / "sift_learn.fvecs")
    query = read_fvecs(root / "sift_query.fvecs")
    ground_truth = read_ivecs(root / "sift_groundtruth.ivecs").astype(np.int64, copy=False)

    vectors = np.vstack([base, learn, query]).astype(np.float32, copy=False)
    split_codes = np.concatenate(
        [
            np.zeros(base.shape[0], dtype=np.int16),
            np.ones(learn.shape[0], dtype=np.int16),
            np.full(query.shape[0], 2, dtype=np.int16),
        ]
    )
    split_names = np.asarray(["base", "learn", "query"])
    database_indices = np.arange(base.shape[0], dtype=np.int64)
    train_indices = np.arange(base.shape[0], base.shape[0] + learn.shape[0], dtype=np.int64)
    query_indices = np.arange(base.shape[0] + learn.shape[0], vectors.shape[0], dtype=np.int64)
    output_path = output_dir / "sift1m.npz"
    save_unified_dataset(
        output_path,
        dataset_name="sift1m",
        family_name="sift",
        unit_type="descriptor",
        vectors=vectors,
        split_codes=split_codes,
        split_names=split_names,
        database_indices=database_indices,
        train_indices=train_indices,
        query_indices=query_indices,
        ground_truth_neighbors=ground_truth,
        metadata={
            "raw_root": str(root),
            "official_split_sizes": {
                "base": int(base.shape[0]),
                "learn": int(learn.shape[0]),
                "query": int(query.shape[0]),
            },
            "ground_truth_k": int(ground_truth.shape[1]) if ground_truth.ndim == 2 else 0,
        },
    )
    return {
        "dataset_name": "sift1m",
        "output_path": str(output_path),
        "num_samples": int(vectors.shape[0]),
        "feature_dim": int(vectors.shape[1]),
        "database_size": int(database_indices.size),
        "query_size": int(query_indices.size),
        "train_size": int(train_indices.size),
    }


def export_cshsi_scene(raw_root: Path, output_dir: Path, scene_name: str) -> dict:
    scene = load_cshsi_scene(raw_root, scene_name)
    vectors, labels, coords = flatten_hsi_cube(scene.cube, scene.labels, labeled_only=False)
    class_ids = np.asarray(sorted(int(x) for x in np.unique(labels) if x != 0), dtype=np.int32)
    output_path = output_dir / f"cshsi_{scene_name}.npz"
    save_unified_dataset(
        output_path,
        dataset_name=f"cshsi_{scene_name}",
        family_name="cshsi",
        unit_type="pixel",
        vectors=vectors,
        labels=labels,
        split_codes=np.zeros(vectors.shape[0], dtype=np.int16),
        split_names=np.asarray(["all"]),
        database_indices=np.arange(vectors.shape[0], dtype=np.int64),
        coords=coords,
        class_ids=class_ids,
        metadata={
            "scene_name": scene_name,
            "image_shape": list(scene.cube.shape[:2]),
            "num_bands": int(scene.cube.shape[2]),
            "labeled_pixels": int((labels != 0).sum()),
            "background_label": 0,
        },
    )
    return {
        "dataset_name": f"cshsi_{scene_name}",
        "output_path": str(output_path),
        "num_samples": int(vectors.shape[0]),
        "feature_dim": int(vectors.shape[1]),
        "database_size": int(vectors.shape[0]),
        "query_size": 0,
        "num_classes": int(class_ids.size),
    }


def export_mars(raw_root: Path, output_dir: Path, band_index: int) -> dict:
    image_dir = mars_root(raw_root)
    vectors, names = load_mars_band_vectors(image_dir=image_dir, band_index=band_index)
    coords = np.asarray([parse_mars_tile_name(name) for name in names], dtype=np.int32)
    output_path = output_dir / f"mars_15d_band{band_index + 1}.npz"
    save_unified_dataset(
        output_path,
        dataset_name=f"mars_15d_band{band_index + 1}",
        family_name="mars",
        unit_type="tile",
        vectors=vectors,
        split_codes=np.zeros(vectors.shape[0], dtype=np.int16),
        split_names=np.asarray(["corpus"]),
        database_indices=np.arange(vectors.shape[0], dtype=np.int64),
        coords=coords,
        sample_names=np.asarray(names),
        sample_shape=np.asarray([128, 128], dtype=np.int32),
        metadata={
            "image_dir": str(image_dir),
            "source_num_bands": 15,
            "selected_band_index": int(band_index),
            "labels_available": False,
        },
    )
    return {
        "dataset_name": f"mars_15d_band{band_index + 1}",
        "output_path": str(output_path),
        "num_samples": int(vectors.shape[0]),
        "feature_dim": int(vectors.shape[1]),
        "database_size": int(vectors.shape[0]),
        "query_size": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Root directory containing raw datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "data_processed",
        help="Directory where unified per-dataset files will be stored.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help="Datasets to export. Use 'all' for every supported dataset.",
    )
    parser.add_argument(
        "--mars-band-index",
        type=int,
        default=3,
        help="0-based Mars spectral band index to keep.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Optional JSON manifest path. Defaults to <output-dir>/manifest.json.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        args.manifest_path.resolve()
        if args.manifest_path is not None
        else output_dir / "manifest.json"
    )

    selected = set(args.datasets)
    if "all" in selected:
        selected = {"mnist", "glove", "sift1m", "mars", "cshsi"}

    summaries: list[dict] = []
    if "mnist" in selected:
        print("[info] exporting mnist")
        summaries.append(export_mnist(raw_root, output_dir))
    if "glove" in selected:
        print("[info] exporting glove")
        summaries.append(export_glove(raw_root, output_dir))
    if "sift1m" in selected:
        print("[info] exporting sift1m")
        summaries.append(export_sift1m(raw_root, output_dir))
    if "cshsi" in selected:
        print("[info] exporting cshsi scenes")
        for scene_name in sorted(list_cshsi_scenes(raw_root)):
            print(f"[info] exporting cshsi scene={scene_name}")
            summaries.append(export_cshsi_scene(raw_root, output_dir, scene_name))
    if "mars" in selected:
        print("[info] exporting mars")
        summaries.append(export_mars(raw_root, output_dir, args.mars_band_index))

    manifest = {
        "schema_version": 1,
        "output_dir": str(output_dir),
        "num_files": len(summaries),
        "datasets": summaries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[done] wrote manifest to {manifest_path}")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
