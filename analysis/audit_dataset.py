#!/usr/bin/env python3
"""Read-only integrity and reproducibility audit for processed_325_nodules_v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def patient_split_map(patient_ids: list[str], seed: int = 42) -> dict[str, str]:
    patients = sorted(set(patient_ids))
    rng = random.Random(seed)
    rng.shuffle(patients)
    n_train = int(round(0.70 * len(patients)))
    n_val = int(round(0.15 * len(patients)))
    return {
        patient: ("train" if i < n_train else "val" if i < n_train + n_val else "test")
        for i, patient in enumerate(patients)
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_png_volume(images_dir: Path) -> np.ndarray:
    def key(path: Path):
        digits = "".join(c for c in path.stem if c.isdigit())
        return int(digits) if digits else path.name

    paths = sorted(images_dir.glob("*.png"), key=key)
    arrays = [np.asarray(Image.open(path).convert("F"), dtype=np.float32) for path in paths]
    return np.stack(arrays, axis=0)


def crop_center_pad(volume: np.ndarray, bbox: list[int], pad_value: float = 0) -> np.ndarray:
    y0, y1, x0, x1 = map(int, bbox)
    depth, height, width = volume.shape
    crop = volume[:, y0:y1, x0:x1]
    out = np.full((depth, height, width), pad_value, dtype=volume.dtype)
    yoff = (height - crop.shape[1]) // 2
    xoff = (width - crop.shape[2]) // 2
    out[:, yoff : yoff + crop.shape[1], xoff : xoff + crop.shape[2]] = crop
    return out


def nested_keys(obj, prefix=""):
    found = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            found.append(name)
            found.extend(nested_keys(value, name))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            found.extend(nested_keys(value, f"{prefix}[{i}]"))
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted(p for p in root.glob("nodule_*") if p.is_dir())
    metadata_records = []
    for case_dir in case_dirs:
        metadata_records.append(json.loads((case_dir / "metadata.json").read_text()))
    split_by_patient = patient_split_map([str(m.get("patient_id", "")) for m in metadata_records])

    rows = []
    failures: list[dict] = []
    all_metadata_keys = Counter()
    image_hashes: dict[str, list[str]] = {}
    mask_hashes: dict[str, list[str]] = {}

    for case_dir, metadata in zip(case_dirs, metadata_records):
        case_id = case_dir.name
        patient_id = str(metadata.get("patient_id", ""))
        split = split_by_patient.get(patient_id, "")
        for key in nested_keys(metadata):
            all_metadata_keys[key] += 1

        required = [
            case_dir / "image_volume.npy",
            case_dir / "mask_consensus.npy",
            case_dir / "mask_union.npy",
            case_dir / "metadata.json",
        ]
        missing = [p.name for p in required if not p.exists()]
        if missing:
            failures.append({"case": case_id, "check": "required_files", "detail": missing})
            continue

        image = np.load(case_dir / "image_volume.npy")
        consensus = np.load(case_dir / "mask_consensus.npy")
        union = np.load(case_dir / "mask_union.npy")
        reader_paths = sorted(case_dir.glob("mask_[0-3].npy"))
        readers = [np.load(path) for path in reader_paths]
        reader_stack = np.stack([(m > 0).astype(np.uint8) for m in readers], axis=0)
        expected_union = reader_stack.any(axis=0).astype(np.uint8)
        expected_consensus = (reader_stack.mean(axis=0) >= 0.5).astype(np.uint8)

        shape_ok = image.shape == consensus.shape == union.shape and all(m.shape == image.shape for m in readers)
        finite = bool(np.isfinite(image).all())
        intensity_ok = bool(finite and image.min() >= -1e-7 and image.max() <= 1 + 1e-7)
        masks_binary = bool(
            set(np.unique(consensus)).issubset({0, 1})
            and set(np.unique(union)).issubset({0, 1})
            and all(set(np.unique(m)).issubset({0, 1}) for m in readers)
        )
        union_exact = bool(np.array_equal(union, expected_union))
        consensus_exact = bool(np.array_equal(consensus, expected_consensus))

        bbox = metadata.get("preprocess", {}).get("crop_bbox_y0_y1_x0_x1")
        original_repro_max_abs = np.nan
        reader_repro_exact = True
        if bbox and (case_dir / "images").exists():
            raw = load_png_volume(case_dir / "images")
            intensity = metadata.get("preprocess", {}).get("intensity", {})
            low = float(intensity.get("train_low", 0.0))
            high = float(intensity.get("train_high", 1.0))
            normalized = np.clip((raw - low) / (high - low + 1e-8), 0.0, 1.0).astype(np.float32)
            reproduced = crop_center_pad(normalized, bbox, 0.0)
            original_repro_max_abs = float(np.max(np.abs(reproduced - image)))
            for aligned_path in reader_paths:
                original_path = case_dir / "original_masks" / aligned_path.name
                if not original_path.exists():
                    reader_repro_exact = False
                    continue
                original = (np.load(original_path) > 0).astype(np.uint8)
                aligned_expected = crop_center_pad(original, bbox, 0)
                reader_repro_exact = reader_repro_exact and bool(
                    np.array_equal(aligned_expected, np.load(aligned_path))
                )

        union_coords = np.argwhere(union > 0)
        if union_coords.size:
            centroid = union_coords.mean(axis=0)
            center_distance_px = float(
                np.sqrt((centroid[1] - (image.shape[1] - 1) / 2) ** 2 + (centroid[2] - (image.shape[2] - 1) / 2) ** 2)
            )
        else:
            center_distance_px = np.nan

        vote_fraction = reader_stack.mean(axis=0)
        disagreement = ((vote_fraction > 0) & (vote_fraction < 1)).sum()
        mask_slices = int((consensus.reshape(consensus.shape[0], -1).sum(axis=1) > 0).sum())
        empty_slices = int(consensus.shape[0] - mask_slices)
        image_hash = sha256_file(case_dir / "image_volume.npy")
        mask_hash = sha256_file(case_dir / "mask_consensus.npy")
        image_hashes.setdefault(image_hash, []).append(case_id)
        mask_hashes.setdefault(mask_hash, []).append(case_id)

        record = {
            "subset_nodule_id": case_id,
            "patient_id": patient_id,
            "split": split,
            "native_nodule_id": metadata.get("native_nodule_id"),
            "series_instance_uid": metadata.get("SeriesInstanceUID"),
            "depth": int(image.shape[0]),
            "height": int(image.shape[1]),
            "width": int(image.shape[2]),
            "image_dtype": str(image.dtype),
            "image_min": float(image.min()),
            "image_max": float(image.max()),
            "image_mean": float(image.mean()),
            "n_readers": len(readers),
            "consensus_voxels": int(consensus.sum()),
            "union_voxels": int(union.sum()),
            "disagreement_voxels": int(disagreement),
            "disagreement_fraction_within_union": float(disagreement / max(int(union.sum()), 1)),
            "positive_slices": mask_slices,
            "empty_slices": empty_slices,
            "center_distance_px": center_distance_px,
            "shape_ok": shape_ok,
            "finite": finite,
            "intensity_01": intensity_ok,
            "masks_binary": masks_binary,
            "union_exact": union_exact,
            "consensus_exact": consensus_exact,
            "reader_alignment_reproducible": reader_repro_exact,
            "image_preprocessing_max_abs_error": original_repro_max_abs,
            "image_sha256": image_hash,
            "consensus_sha256": mask_hash,
        }
        rows.append(record)
        for check in [
            "shape_ok",
            "finite",
            "intensity_01",
            "masks_binary",
            "union_exact",
            "consensus_exact",
            "reader_alignment_reproducible",
        ]:
            if not record[check]:
                failures.append({"case": case_id, "check": check, "detail": "failed"})
        if np.isfinite(original_repro_max_abs) and original_repro_max_abs > 1e-6:
            failures.append(
                {"case": case_id, "check": "image_preprocessing_reproduction", "detail": original_repro_max_abs}
            )

    frame = pd.DataFrame(rows).sort_values("subset_nodule_id")
    frame.to_csv(out_dir / "final_325_manifest_audit.csv", index=False)
    pd.DataFrame(failures).to_csv(out_dir / "audit_failures.csv", index=False)
    duplicate_images = {k: v for k, v in image_hashes.items() if len(v) > 1}
    duplicate_masks = {k: v for k, v in mask_hashes.items() if len(v) > 1}
    malignancy_keys = sorted(k for k in all_metadata_keys if "malignan" in k.lower() or "rating" in k.lower())

    summary = {
        "dataset_root": str(root),
        "n_case_directories": len(case_dirs),
        "n_audited_cases": int(len(frame)),
        "n_patients": int(frame["patient_id"].nunique()) if len(frame) else 0,
        "n_failures": len(failures),
        "split_cases": frame["split"].value_counts().to_dict(),
        "split_patients": frame.groupby("split")["patient_id"].nunique().to_dict(),
        "depth_distribution": frame["depth"].value_counts().sort_index().to_dict(),
        "reader_count_distribution": frame["n_readers"].value_counts().sort_index().to_dict(),
        "total_slices": int(frame["depth"].sum()),
        "positive_slices": int(frame["positive_slices"].sum()),
        "empty_slices": int(frame["empty_slices"].sum()),
        "total_consensus_voxels": int(frame["consensus_voxels"].sum()),
        "total_union_voxels": int(frame["union_voxels"].sum()),
        "median_disagreement_fraction_within_union": float(frame["disagreement_fraction_within_union"].median()),
        "cases_with_zero_consensus": int((frame["consensus_voxels"] == 0).sum()),
        "cases_with_zero_union": int((frame["union_voxels"] == 0).sum()),
        "duplicate_image_groups": duplicate_images,
        "duplicate_consensus_mask_groups": duplicate_masks,
        "malignancy_or_rating_metadata_keys": malignancy_keys,
        "all_checks_pass": len(failures) == 0,
    }
    (out_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
