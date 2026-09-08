#!/usr/bin/env python3
"""Build the slice-aligned, multi-reader v3 candidate-ROI dataset.

This script reads Isha's supplied processed_325_nodules_v2 folders but rebuilds
the model inputs from the preserved original PNGs and original reader masks.
It intentionally does not use a ground-truth mask to crop or center the input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn import __version__ as sklearn_version
from sklearn.model_selection import StratifiedGroupKFold


def natural_key(path: Path) -> tuple:
    parts = re.split(r"(\d+)", path.name)
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def patient_split_map(patient_ids: list[str], seed: int) -> dict[str, str]:
    patients = sorted(set(patient_ids))
    rng = random.Random(seed)
    rng.shuffle(patients)
    n_train = int(round(0.70 * len(patients)))
    n_val = int(round(0.15 * len(patients)))
    return {
        patient: ("train" if i < n_train else "val" if i < n_train + n_val else "test")
        for i, patient in enumerate(patients)
    }


def load_png_volume(images_dir: Path) -> np.ndarray:
    paths = sorted(images_dir.glob("*.png"), key=natural_key)
    if not paths:
        raise RuntimeError(f"No PNG slices in {images_dir}")
    return np.stack(
        [np.asarray(Image.open(path).convert("F"), dtype=np.float32) for path in paths],
        axis=0,
    )


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def normalized_entropy(vote_fraction: np.ndarray) -> np.ndarray:
    p = np.clip(vote_fraction.astype(np.float32), 1e-6, 1.0 - 1e-6)
    entropy = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
    entropy[(vote_fraction <= 0) | (vote_fraction >= 1)] = 0.0
    return entropy.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/processed_325_nodules_v2"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/final_325_nodules_v3"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    cases_out = output / "cases"
    manifest_path = output / "final_325_manifest.csv"
    if manifest_path.exists() and not args.force:
        raise SystemExit(f"{manifest_path} already exists; use --force to rebuild")
    validation_path = output / "validation_report.json"
    if args.force and validation_path.exists():
        # A forced rebuild invalidates every prior PASS fingerprint immediately.
        # Training remains blocked until script 20 completes again.
        validation_path.unlink()
    output.mkdir(parents=True, exist_ok=True)
    cases_out.mkdir(parents=True, exist_ok=True)

    source_cases = sorted(
        [path for path in source.glob("nodule_*") if path.is_dir()],
        key=natural_key,
    )
    if len(source_cases) != 325:
        raise RuntimeError(f"Expected 325 source cases, found {len(source_cases)}")

    metadata_by_case = {
        case.name: json.loads((case / "metadata.json").read_text()) for case in source_cases
    }
    split_map = patient_split_map(
        [str(meta["patient_id"]) for meta in metadata_by_case.values()], args.seed
    )

    manifest_rows = []
    slice_rows = []
    seen_image_hashes: dict[str, str] = {}
    for source_case in source_cases:
        case_id = source_case.name
        source_meta = metadata_by_case[case_id]
        patient_id = str(source_meta["patient_id"])
        split = split_map[patient_id]
        raw_image = load_png_volume(source_case / "images")
        image = np.clip(np.rint(raw_image), 0, 255).astype(np.uint8)

        original_mask_dir = source_case / "original_masks"
        mask_arrays = []
        contour_present = []
        for reader_index in range(4):
            path = original_mask_dir / f"mask_{reader_index}.npy"
            if not path.exists():
                mask = np.zeros(image.shape, dtype=np.uint8)
                is_present = False
            else:
                mask = (np.load(path) > 0).astype(np.uint8)
                if mask.shape != image.shape:
                    raise RuntimeError(
                        f"{case_id} reader {reader_index}: image {image.shape}, mask {mask.shape}"
                    )
                is_present = bool(mask.any())
            mask_arrays.append(mask)
            contour_present.append(is_present)
        reader_masks = np.stack(mask_arrays, axis=0).astype(np.uint8)
        contour_present_array = np.asarray(contour_present, dtype=np.uint8)
        n_contours = int(contour_present_array.sum())
        if n_contours < 1:
            raise RuntimeError(f"{case_id} has no nodule contour")

        # The source archive describes mask-0 through mask-3 as the first
        # through fourth clustered annotation.  LIDC-IDRI used four reader
        # review sessions per scan; an absent higher contour is therefore zero
        # reader support for this cluster, not an unavailable reader.  Slot
        # identity is not retained, but the total support count is.
        review_sessions = 4
        vote_count = reader_masks.sum(axis=0).astype(np.uint8)
        vote_fraction = (vote_count / float(review_sessions)).astype(np.float32)
        union = (vote_count >= 1).astype(np.uint8)
        majority = (vote_count >= 2).astype(np.uint8)
        strict_majority = (vote_count >= 3).astype(np.uint8)
        unanimous = (vote_count >= 4).astype(np.uint8)
        entropy = normalized_entropy(vote_fraction)

        destination = cases_out / case_id
        destination.mkdir(parents=True, exist_ok=True)
        np.save(destination / "image_volume_uint8.npy", image)
        stale_normalized_image = destination / "image_volume.npy"
        if stale_normalized_image.exists():
            stale_normalized_image.unlink()
        np.save(destination / "reader_masks.npy", reader_masks)
        np.save(destination / "reader_contour_present.npy", contour_present_array)
        stale_availability = destination / "reader_available.npy"
        if stale_availability.exists():
            stale_availability.unlink()
        np.save(destination / "reader_vote_count.npy", vote_count)
        np.save(destination / "reader_vote_fraction.npy", vote_fraction)
        np.save(destination / "mask_union.npy", union)
        np.save(destination / "mask_majority.npy", majority)
        np.save(destination / "mask_strict_majority.npy", strict_majority)
        np.save(destination / "mask_unanimous.npy", unanimous)
        np.save(destination / "annotation_entropy.npy", entropy)

        case_metadata = {
            "dataset_version": "v3.1",
            "subset_nodule_id": case_id,
            "patient_id": patient_id,
            "native_nodule_id": source_meta.get("native_nodule_id"),
            "SeriesInstanceUID": source_meta.get("SeriesInstanceUID"),
            "split": split,
            "source": {
                "dataset": "LIDC-IDRI nodule crop subset",
                "input_folder": "preserved original PNGs from Isha's v2 package",
                "source_dataset_version": source_meta.get("dataset_version", "v2"),
                "task_scope": "segmentation inside the supplied candidate-centered ROI",
                "candidate_selection": (
                    "inherited from the supplied source package; this dataset does not "
                    "constitute full-scan nodule detection"
                ),
            },
            "preprocessing": {
                "slice_order": "natural numeric order",
                "intensity": (
                    "raw uint8 PNG values preserved; percentile normalization is "
                    "fitted inside each experiment's optimization fold"
                ),
                "ground_truth_mask_used_to_transform_input": False,
                "spatial_crop_or_recenter": False,
            },
            "annotations": {
                "reader_slots": 4,
                "review_sessions": review_sessions,
                "clustered_contour_slots": [
                    i for i, present in enumerate(contour_present) if present
                ],
                "n_clustered_contours": n_contours,
                "slot_identity": "ordered clustered annotations; radiologist identity not retained",
                "vote_fraction_denominator": "four LIDC-IDRI reader review sessions",
            },
            "shape_dhw": list(image.shape),
        }
        (destination / "metadata.json").write_text(
            json.dumps(case_metadata, indent=2, sort_keys=True) + "\n"
        )

        relative = destination.relative_to(output)
        image_hash = sha256_array(image)
        if image_hash in seen_image_hashes:
            raise RuntimeError(
                f"Duplicate normalized image: {case_id} and {seen_image_hashes[image_hash]}"
            )
        seen_image_hashes[image_hash] = case_id
        disagreement_voxels = int(((vote_fraction > 0) & (vote_fraction < 1)).sum())
        row = {
            "subset_nodule_id": case_id,
            "patient_id": patient_id,
            "split": split,
            "cv_fold": -1,
            "native_nodule_id": source_meta.get("native_nodule_id"),
            "series_instance_uid": source_meta.get("SeriesInstanceUID"),
            "depth": int(image.shape[0]),
            "height": int(image.shape[1]),
            "width": int(image.shape[2]),
            "n_review_sessions": review_sessions,
            "n_clustered_contours": n_contours,
            "clustered_contour_slots": "|".join(
                str(i) for i, x in enumerate(contour_present) if x
            ),
            "union_voxels": int(union.sum()),
            "majority_voxels": int(majority.sum()),
            "strict_majority_voxels": int(strict_majority.sum()),
            "unanimous_voxels": int(unanimous.sum()),
            "disagreement_voxels": disagreement_voxels,
            "disagreement_fraction_within_union": float(disagreement_voxels / max(int(union.sum()), 1)),
            "image_min": float(image.min()),
            "image_max": float(image.max()),
            "image_mean": float(image.mean()),
            "image_sha256": image_hash,
            "image_volume_uint8_path": str(relative / "image_volume_uint8.npy"),
            "reader_masks_path": str(relative / "reader_masks.npy"),
            "reader_contour_present_path": str(relative / "reader_contour_present.npy"),
            "reader_vote_count_path": str(relative / "reader_vote_count.npy"),
            "reader_vote_fraction_path": str(relative / "reader_vote_fraction.npy"),
            "mask_union_path": str(relative / "mask_union.npy"),
            "mask_majority_path": str(relative / "mask_majority.npy"),
            "mask_strict_majority_path": str(relative / "mask_strict_majority.npy"),
            "mask_unanimous_path": str(relative / "mask_unanimous.npy"),
            "annotation_entropy_path": str(relative / "annotation_entropy.npy"),
            "metadata_path": str(relative / "metadata.json"),
        }
        manifest_rows.append(row)
        for slice_index in range(image.shape[0]):
            slice_vote = vote_fraction[slice_index]
            slice_rows.append(
                {
                    "subset_nodule_id": case_id,
                    "patient_id": patient_id,
                    "split": split,
                    "cv_fold": -1,
                    "slice_index": slice_index,
                    "union_voxels": int(union[slice_index].sum()),
                    "majority_voxels": int(majority[slice_index].sum()),
                    "strict_majority_voxels": int(strict_majority[slice_index].sum()),
                    "unanimous_voxels": int(unanimous[slice_index].sum()),
                    "disagreement_voxels": int(
                        ((slice_vote > 0) & (slice_vote < 1)).sum()
                    ),
                }
            )

    manifest = pd.DataFrame(manifest_rows).sort_values("subset_nodule_id").reset_index(drop=True)
    # Five patient-disjoint outer folds over the full cohort.  These folds are
    # the final comparison protocol because every case receives exactly one
    # out-of-fold prediction.  The legacy train/val/test column remains useful
    # for quick development runs and backwards compatibility.
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    for fold, (_, held_out_index) in enumerate(
        splitter.split(
            manifest,
            y=manifest["n_clustered_contours"],
            groups=manifest["patient_id"],
        )
    ):
        manifest.loc[manifest.iloc[held_out_index].index, "cv_fold"] = fold
    slice_manifest = pd.DataFrame(slice_rows)
    fold_map = manifest.set_index("subset_nodule_id")["cv_fold"].to_dict()
    slice_manifest["cv_fold"] = slice_manifest["subset_nodule_id"].map(fold_map)

    if manifest["patient_id"].nunique() != 247:
        raise RuntimeError("Expected 247 unique patients")
    split_sets = {
        split: set(manifest.loc[manifest["split"] == split, "patient_id"])
        for split in ("train", "val", "test")
    }
    if split_sets["train"] & split_sets["val"] or split_sets["train"] & split_sets["test"] or split_sets["val"] & split_sets["test"]:
        raise RuntimeError("Patient leakage detected across train/val/test")
    if set(manifest["cv_fold"].unique()) != set(range(5)):
        raise RuntimeError("Every case must be assigned to one of five outer CV folds")
    patient_fold_counts = manifest.groupby("patient_id")["cv_fold"].nunique()
    if not bool((patient_fold_counts == 1).all()):
        raise RuntimeError("A patient was assigned to more than one outer CV fold")

    manifest.to_csv(manifest_path, index=False)
    slice_manifest.to_csv(output / "final_2000_slice_manifest.csv", index=False)
    dataset_summary = {
        "dataset_version": "v3.1",
        "n_nodules": int(len(manifest)),
        "n_patients": int(manifest["patient_id"].nunique()),
        "n_slices": int(manifest["depth"].sum()),
        "split_nodules": manifest["split"].value_counts().to_dict(),
        "split_patients": manifest.groupby("split")["patient_id"].nunique().to_dict(),
        "cv_fold_nodules": manifest["cv_fold"].value_counts().sort_index().to_dict(),
        "cv_fold_patients": manifest.groupby("cv_fold")["patient_id"].nunique().to_dict(),
        "clustered_contour_count_distribution": manifest["n_clustered_contours"].value_counts().sort_index().to_dict(),
        "stored_image_dtype": "uint8",
        "normalization_policy": "fit 1st/99th percentiles on each optimization fold only",
        "ground_truth_mask_used_to_transform_input": False,
        "slice_order_repaired": True,
        "splitter": "sklearn.model_selection.StratifiedGroupKFold",
        "scikit_learn_version": sklearn_version,
        "cases_with_nonzero_majority": int((manifest["majority_voxels"] > 0).sum()),
        "target_positive_cases": {
            "T1": int((manifest["union_voxels"] > 0).sum()),
            "T2": int((manifest["majority_voxels"] > 0).sum()),
            "T3": int((manifest["strict_majority_voxels"] > 0).sum()),
            "T4": int((manifest["unanimous_voxels"] > 0).sum()),
        },
        "cases_with_reader_disagreement": int((manifest["disagreement_voxels"] > 0).sum()),
        "task_scope": "segmentation inside the supplied candidate-centered ROI",
    }
    (output / "dataset_summary.json").write_text(
        json.dumps(dataset_summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(dataset_summary, indent=2, sort_keys=True))
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
