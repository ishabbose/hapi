#!/usr/bin/env python3
"""Validate the complete v3 dataset before any model is trained.

The checks here are deliberately strict: a failed assertion stops the run rather
than allowing a malformed image, target, path, or patient split to reach a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "subset_nodule_id",
    "patient_id",
    "split",
    "cv_fold",
    "depth",
    "height",
    "width",
    "n_review_sessions",
    "n_clustered_contours",
    "image_sha256",
    "image_volume_uint8_path",
    "reader_masks_path",
    "reader_contour_present_path",
    "reader_vote_count_path",
    "reader_vote_fraction_path",
    "mask_union_path",
    "mask_majority_path",
    "mask_strict_majority_path",
    "mask_unanimous_path",
    "annotation_entropy_path",
    "metadata_path",
}

ARRAY_PATH_COLUMNS = (
    "image_volume_uint8_path",
    "reader_masks_path",
    "reader_contour_present_path",
    "reader_vote_count_path",
    "reader_vote_fraction_path",
    "mask_union_path",
    "mask_majority_path",
    "mask_strict_majority_path",
    "mask_unanimous_path",
    "annotation_entropy_path",
)


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative_path(root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Path is not portable and relative: {value}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes the dataset root: {value}")
    return resolved


def normalized_entropy(q: np.ndarray) -> np.ndarray:
    clipped = np.clip(q.astype(np.float32), 1e-6, 1.0 - 1e-6)
    entropy = -(clipped * np.log2(clipped) + (1.0 - clipped) * np.log2(1.0 - clipped))
    entropy[(q <= 0) | (q >= 1)] = 0.0
    return entropy.astype(np.float32)


def validate_case(root: Path, row: pd.Series) -> dict[str, object]:
    case_id = str(row["subset_nodule_id"])
    paths = {column: safe_relative_path(root, str(row[column])) for column in ARRAY_PATH_COLUMNS}
    metadata_path = safe_relative_path(root, str(row["metadata_path"]))
    missing = [str(path) for path in [*paths.values(), metadata_path] if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{case_id}: missing files: {missing}")

    image = np.load(paths["image_volume_uint8_path"], allow_pickle=False)
    readers = np.load(paths["reader_masks_path"], allow_pickle=False)
    contour_present = np.load(paths["reader_contour_present_path"], allow_pickle=False)
    votes = np.load(paths["reader_vote_count_path"], allow_pickle=False)
    q = np.load(paths["reader_vote_fraction_path"], allow_pickle=False)
    union = np.load(paths["mask_union_path"], allow_pickle=False)
    majority = np.load(paths["mask_majority_path"], allow_pickle=False)
    strict = np.load(paths["mask_strict_majority_path"], allow_pickle=False)
    unanimous = np.load(paths["mask_unanimous_path"], allow_pickle=False)
    entropy = np.load(paths["annotation_entropy_path"], allow_pickle=False)
    metadata = json.loads(metadata_path.read_text())

    shape = (int(row["depth"]), int(row["height"]), int(row["width"]))
    if image.shape != shape:
        raise ValueError(f"{case_id}: image shape {image.shape} != manifest {shape}")
    if readers.shape != (4, *shape):
        raise ValueError(f"{case_id}: reader shape {readers.shape} != {(4, *shape)}")
    if contour_present.shape != (4,):
        raise ValueError(f"{case_id}: contour-present shape {contour_present.shape} != (4,)")
    for name, array in {
        "votes": votes,
        "q": q,
        "union": union,
        "majority": majority,
        "strict": strict,
        "unanimous": unanimous,
        "entropy": entropy,
    }.items():
        if array.shape != shape:
            raise ValueError(f"{case_id}: {name} shape {array.shape} != {shape}")

    if image.dtype != np.uint8:
        raise ValueError(f"{case_id}: preserved source image must be uint8")
    if sha256_array(image) != str(row["image_sha256"]):
        raise ValueError(f"{case_id}: image hash differs from manifest")

    if not set(np.unique(readers)).issubset({0, 1}):
        raise ValueError(f"{case_id}: reader masks are not binary")
    if not set(np.unique(contour_present)).issubset({0, 1}):
        raise ValueError(f"{case_id}: contour presence is not binary")
    inferred_present = readers.reshape(4, -1).any(axis=1).astype(np.uint8)
    if not np.array_equal(contour_present.astype(np.uint8), inferred_present):
        raise ValueError(f"{case_id}: contour presence disagrees with masks")
    n_contours = int(contour_present.sum())
    if n_contours != int(row["n_clustered_contours"]) or not 1 <= n_contours <= 4:
        raise ValueError(f"{case_id}: invalid clustered-contour count {n_contours}")
    if int(row["n_review_sessions"]) != 4:
        raise ValueError(f"{case_id}: expected four LIDC-IDRI review sessions")

    expected_votes = readers.sum(axis=0).astype(np.uint8)
    expected_q = expected_votes.astype(np.float32) / 4.0
    expected_union = (expected_votes >= 1).astype(np.uint8)
    expected_majority = (expected_votes >= 2).astype(np.uint8)
    expected_strict = (expected_votes >= 3).astype(np.uint8)
    expected_unanimous = (expected_votes >= 4).astype(np.uint8)
    expected_entropy = normalized_entropy(expected_q)

    exact_targets = {
        "vote count": (votes, expected_votes),
        "vote fraction": (q, expected_q),
        "union": (union, expected_union),
        "majority": (majority, expected_majority),
        "strict majority": (strict, expected_strict),
        "unanimous": (unanimous, expected_unanimous),
        "entropy": (entropy, expected_entropy),
    }
    for name, (actual, expected) in exact_targets.items():
        if not np.array_equal(actual, expected):
            raise ValueError(f"{case_id}: {name} is inconsistent with reader masks")

    if int(union.sum()) <= 0:
        raise ValueError(f"{case_id}: T1/union target must be nonempty")
    if metadata.get("subset_nodule_id") != case_id:
        raise ValueError(f"{case_id}: metadata identifier mismatch")
    if metadata.get("patient_id") != str(row["patient_id"]):
        raise ValueError(f"{case_id}: metadata patient mismatch")
    if metadata.get("split") != str(row["split"]):
        raise ValueError(f"{case_id}: metadata split mismatch")
    preprocessing = metadata.get("preprocessing", {})
    if preprocessing.get("ground_truth_mask_used_to_transform_input") is not False:
        raise ValueError(f"{case_id}: input transform is not marked target-independent")
    if preprocessing.get("slice_order") != "natural numeric order":
        raise ValueError(f"{case_id}: numeric z-order is not recorded")

    return {
        "subset_nodule_id": case_id,
        "patient_id": str(row["patient_id"]),
        "split": str(row["split"]),
        "cv_fold": int(row["cv_fold"]),
        "depth": shape[0],
        "n_clustered_contours": n_contours,
        "image_sha256": sha256_array(image),
        "target_sha256": hashlib.sha256(
            np.ascontiguousarray(readers).tobytes()
            + np.ascontiguousarray(votes).tobytes()
        ).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/final_325_nodules_v3"))
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    root = args.data_root.resolve()
    manifest_path = root / "final_325_manifest.csv"
    slice_manifest_path = root / "final_2000_slice_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    missing_columns = sorted(REQUIRED_COLUMNS - set(manifest.columns))
    if missing_columns:
        raise ValueError(f"Manifest is missing columns: {missing_columns}")
    if len(manifest) != 325 or manifest["subset_nodule_id"].nunique() != 325:
        raise ValueError("Manifest must contain exactly 325 unique nodules")
    if manifest["patient_id"].nunique() != 247:
        raise ValueError("Manifest must contain exactly 247 unique patients")
    if set(manifest["split"]) != {"train", "val", "test"}:
        raise ValueError("Expected train, val, and test splits")
    if set(manifest["cv_fold"].astype(int)) != set(range(5)):
        raise ValueError("Expected five outer CV folds numbered 0 through 4")
    if int(manifest.groupby("patient_id")["split"].nunique().max()) != 1:
        raise ValueError("Patient leakage exists across train/val/test")
    if int(manifest.groupby("patient_id")["cv_fold"].nunique().max()) != 1:
        raise ValueError("Patient leakage exists across outer CV folds")

    rows = []
    failures = []
    for _, row in manifest.iterrows():
        try:
            rows.append(validate_case(root, row))
        except Exception as error:  # Aggregate all cases before failing.
            failures.append({"subset_nodule_id": row.get("subset_nodule_id"), "error": str(error)})

    slice_manifest = pd.read_csv(slice_manifest_path)
    expected_slices = int(manifest["depth"].sum())
    if len(slice_manifest) != expected_slices:
        failures.append(
            {"subset_nodule_id": "<slice_manifest>", "error": f"{len(slice_manifest)} rows != {expected_slices}"}
        )
    duplicate_images = len(rows) - len({row["image_sha256"] for row in rows})
    if duplicate_images:
        failures.append(
            {"subset_nodule_id": "<dataset>", "error": f"{duplicate_images} duplicate image hashes"}
        )

    ordered_hashes = "".join(
        row["image_sha256"] + row["target_sha256"]
        for row in sorted(rows, key=lambda item: item["subset_nodule_id"])
    )
    report = {
        "status": "PASS" if not failures else "FAIL",
        "dataset_version": "v3.1",
        "n_cases_checked": len(rows),
        "n_failures": len(failures),
        "n_patients": int(manifest["patient_id"].nunique()),
        "n_slices": expected_slices,
        "split_nodules": manifest["split"].value_counts().sort_index().to_dict(),
        "split_patients": manifest.groupby("split")["patient_id"].nunique().to_dict(),
        "cv_fold_nodules": manifest["cv_fold"].value_counts().sort_index().to_dict(),
        "cv_fold_patients": manifest.groupby("cv_fold")["patient_id"].nunique().to_dict(),
        "clustered_contour_count_distribution": manifest["n_clustered_contours"].value_counts().sort_index().to_dict(),
        "dataset_content_sha256": hashlib.sha256(ordered_hashes.encode("ascii")).hexdigest(),
        "manifest_sha256": sha256_file(manifest_path),
        "slice_manifest_sha256": sha256_file(slice_manifest_path),
        "failures": failures,
    }
    report_path = args.report or root / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
