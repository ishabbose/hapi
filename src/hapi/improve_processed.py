"""Build processed_325_nodules_v2: train-only intensity, crop/pad, consensus/union."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from hapi.config import EXPECTED_ELIGIBLE_NODULES
from hapi.preprocess import load_png

DATASET_VERSION = "v2"
CONSENSUS_THRESHOLD = 0.5
INTENSITY_LOW_P = 1.0
INTENSITY_HIGH_P = 99.0
CROP_MARGIN_PX = 4


def _image_files_from_row(row: pd.Series) -> list[str]:
    if "image_files" in row.index and pd.notna(row["image_files"]):
        return [x for x in str(row["image_files"]).split("|") if x]
    images_dir = Path(str(row.get("images_dir", "")))
    return sorted(str(p) for p in images_dir.glob("*.png"))


def load_reader_masks(row: pd.Series, n_slices: int, height: int, width: int):
    masks: list[np.ndarray | None] = []
    for mask_number in range(4):
        column = f"mask_{mask_number}_path"
        raw = ""
        if column in row.index and pd.notna(row[column]):
            raw = str(row[column]).strip()
        if raw in {"", "nan"} or not Path(raw).exists():
            masks.append(None)
            continue
        volume = np.load(raw)
        binary = (np.asarray(volume) > 0).astype(np.uint8)
        if binary.ndim != 3:
            masks.append(None)
            continue
        if binary.shape[0] != n_slices or binary.shape[-2:] != (height, width):
            aligned = np.zeros((n_slices, height, width), dtype=np.uint8)
            d = min(n_slices, binary.shape[0])
            h = min(height, binary.shape[1])
            w = min(width, binary.shape[2])
            aligned[:d, :h, :w] = binary[:d, :h, :w]
            masks.append(aligned)
        else:
            masks.append(binary)
    return masks


def available_masks(masks: list[np.ndarray | None]) -> list[np.ndarray]:
    return [m for m in masks if m is not None]


def union_mask(masks: list[np.ndarray], shape: tuple[int, int, int]) -> np.ndarray:
    if not masks:
        return np.zeros(shape, dtype=np.uint8)
    stacked = np.stack(masks, axis=0)
    return (stacked > 0).any(axis=0).astype(np.uint8)


def consensus_mask(
    masks: list[np.ndarray],
    shape: tuple[int, int, int],
    threshold: float = CONSENSUS_THRESHOLD,
) -> np.ndarray:
    if not masks:
        return np.zeros(shape, dtype=np.uint8)
    stacked = np.stack(masks, axis=0).astype(np.float32)
    return (stacked.mean(axis=0) >= threshold).astype(np.uint8)


def inplane_bbox(foreground: np.ndarray, margin: int, height: int, width: int):
    projection = foreground.any(axis=0)
    coords = np.argwhere(projection)
    if coords.size == 0:
        return 0, height, 0, width, False
    y0 = int(max(0, coords[:, 0].min() - margin))
    y1 = int(min(height, coords[:, 0].max() + 1 + margin))
    x0 = int(max(0, coords[:, 1].min() - margin))
    x1 = int(min(width, coords[:, 1].max() + 1 + margin))
    cropped = not (y0 == 0 and y1 == height and x0 == 0 and x1 == width)
    return y0, y1, x0, x1, cropped


def center_pad_inplane(volume: np.ndarray, y0: int, y1: int, x0: int, x1: int, pad_value):
    depth, height, width = volume.shape
    crop = volume[:, y0:y1, x0:x1]
    crop_h, crop_w = crop.shape[1], crop.shape[2]
    out = np.full((depth, height, width), pad_value, dtype=volume.dtype)
    y_off = (height - crop_h) // 2
    x_off = (width - crop_w) // 2
    out[:, y_off : y_off + crop_h, x_off : x_off + crop_w] = crop
    return out, int(y_off), int(x_off)


def apply_train_intensity(volume: np.ndarray, low: float, high: float) -> np.ndarray:
    if high <= low:
        denom = float(volume.max() - volume.min()) + 1e-8
        return np.clip((volume - float(volume.min())) / denom, 0.0, 1.0).astype(np.float32)
    return np.clip((volume - low) / (high - low + 1e-8), 0.0, 1.0).astype(np.float32)


def fit_train_intensity(
    rows: pd.DataFrame,
    volumes: dict[str, np.ndarray],
    low_p: float = INTENSITY_LOW_P,
    high_p: float = INTENSITY_HIGH_P,
) -> tuple[float, float]:
    train_ids = rows.loc[rows["split"].astype(str) == "train", "subset_nodule_id"].astype(str)
    chunks = [volumes[str(i)].ravel() for i in train_ids if str(i) in volumes]
    if not chunks:
        raise RuntimeError("No train volumes available to fit intensity stats.")
    pixels = np.concatenate(chunks)
    low = float(np.percentile(pixels, low_p))
    high = float(np.percentile(pixels, high_p))
    return low, high


def copy_original_assets(row: pd.Series, image_files: list[str], dest: Path) -> None:
    images_dir = dest / "images"
    original_masks_dir = dest / "original_masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    original_masks_dir.mkdir(parents=True, exist_ok=True)
    for src in image_files:
        src_path = Path(src)
        if src_path.exists():
            shutil.copy2(src_path, images_dir / src_path.name)
    for mask_number in range(4):
        column = f"mask_{mask_number}_path"
        if column not in row.index or pd.isna(row[column]):
            continue
        src_path = Path(str(row[column]))
        if src_path.exists():
            shutil.copy2(src_path, original_masks_dir / f"mask_{mask_number}.npy")


def qc_row(
    *,
    original_shape: tuple[int, ...],
    image_shape: tuple[int, ...],
    n_slices_expected: int,
    split_original: str,
    split_out: str,
    consensus: np.ndarray,
    union: np.ndarray,
    image: np.ndarray,
    n_readers: int,
) -> dict:
    checks = {
        "qc_shape_match": list(original_shape) == list(image_shape),
        "qc_slice_count_match": int(image_shape[0]) == int(n_slices_expected),
        "qc_split_preserved": str(split_original) == str(split_out),
        "qc_no_nan": bool(np.isfinite(image).all()),
        "qc_intensity_01": bool(image.min() >= -1e-6 and image.max() <= 1.0 + 1e-6),
        "qc_consensus_subset_union": bool(np.all(consensus <= union)),
        "qc_n_readers": int(n_readers),
    }
    checks["qc_pass"] = all(
        checks[k]
        for k in (
            "qc_shape_match",
            "qc_slice_count_match",
            "qc_split_preserved",
            "qc_no_nan",
            "qc_intensity_01",
            "qc_consensus_subset_union",
        )
    )
    return checks


def build_improved_processed_v2(
    manifest: pd.DataFrame,
    output_root: Path,
    source_processed_dir: Path,
    *,
    low_p: float = INTENSITY_LOW_P,
    high_p: float = INTENSITY_HIGH_P,
    crop_margin_px: int = CROP_MARGIN_PX,
    consensus_threshold: float = CONSENSUS_THRESHOLD,
) -> tuple[pd.DataFrame, dict]:
    if "split" not in manifest.columns:
        raise ValueError("Manifest is missing split. Re-run 10 then 11.")
    if len(manifest) != EXPECTED_ELIGIBLE_NODULES:
        raise ValueError(
            f"Expected {EXPECTED_ELIGIBLE_NODULES} nodules, got {len(manifest)}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    raw_volumes: dict[str, np.ndarray] = {}
    loaded_rows: list[tuple[pd.Series, list[str], np.ndarray]] = []

    for _, row in manifest.iterrows():
        subset_id = str(row["subset_nodule_id"])
        image_files = _image_files_from_row(row)
        if not image_files:
            raise RuntimeError(f"No images for {subset_id}")
        arrays = [load_png(path) for path in image_files]
        volume = np.stack(arrays, axis=0).astype(np.float32)
        raw_volumes[subset_id] = volume
        loaded_rows.append((row, image_files, volume))

    train_low, train_high = fit_train_intensity(manifest, raw_volumes, low_p, high_p)
    params = {
        "dataset_version": DATASET_VERSION,
        "n_nodules": int(len(manifest)),
        "source_processed_dir": str(source_processed_dir),
        "intensity": {
            "method": "train_percentile_clip_minmax",
            "low_percentile": low_p,
            "high_percentile": high_p,
            "fitted_on_split": "train",
            "train_n_volumes": int((manifest["split"].astype(str) == "train").sum()),
            "train_low": train_low,
            "train_high": train_high,
        },
        "spatial": {
            "mode": "union_mask_inplane_crop_center_pad",
            "crop_margin_px": crop_margin_px,
            "preserve_depth": True,
            "preserve_hw": True,
            "preserve_slice_order": True,
        },
        "masks": {
            "consensus_threshold": consensus_threshold,
            "original_copied_to": "original_masks/",
            "aligned_reader_masks": "mask_0.npy … mask_3.npy",
            "consensus_name": "mask_consensus.npy",
            "union_name": "mask_union.npy",
        },
    }

    out_rows = []
    for row, image_files, raw in loaded_rows:
        subset_id = str(row["subset_nodule_id"])
        dest = output_root / subset_id
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True, exist_ok=True)
        copy_original_assets(row, image_files, dest)

        depth, height, width = raw.shape
        original_shape = (int(depth), int(height), int(width))
        readers = load_reader_masks(row, depth, height, width)
        present = available_masks(readers)
        union = union_mask(present, raw.shape)
        consensus = consensus_mask(present, raw.shape, consensus_threshold)

        y0, y1, x0, x1, did_crop = inplane_bbox(union, crop_margin_px, height, width)
        image_norm = apply_train_intensity(raw, train_low, train_high)
        image_out, y_off, x_off = center_pad_inplane(image_norm, y0, y1, x0, x1, 0.0)
        union_out, _, _ = center_pad_inplane(union, y0, y1, x0, x1, 0)
        consensus_out, _, _ = center_pad_inplane(consensus, y0, y1, x0, x1, 0)

        aligned_paths = {}
        for mask_number, mask in enumerate(readers):
            out_mask_path = dest / f"mask_{mask_number}.npy"
            if mask is None:
                aligned_paths[f"mask_{mask_number}_path"] = ""
                continue
            aligned, _, _ = center_pad_inplane(mask, y0, y1, x0, x1, 0)
            np.save(out_mask_path, aligned.astype(np.uint8))
            aligned_paths[f"mask_{mask_number}_path"] = str(out_mask_path)

        np.save(dest / "image_volume.npy", image_out.astype(np.float32))
        np.save(dest / "mask_consensus.npy", consensus_out.astype(np.uint8))
        np.save(dest / "mask_union.npy", union_out.astype(np.uint8))

        src_meta = source_processed_dir / subset_id / "metadata.json"
        metadata = json.loads(src_meta.read_text()) if src_meta.exists() else {}
        metadata["dataset_version"] = DATASET_VERSION
        metadata["preprocess"] = {
            **params,
            "crop_bbox_y0_y1_x0_x1": [y0, y1, x0, x1],
            "pad_offset_y_x": [y_off, x_off],
            "foreground_crop_applied": did_crop,
            "original_shape": list(original_shape),
            "output_shape": list(image_out.shape),
        }
        (dest / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

        checks = qc_row(
            original_shape=original_shape,
            image_shape=tuple(image_out.shape),
            n_slices_expected=len(image_files),
            split_original=str(row["split"]),
            split_out=str(row["split"]),
            consensus=consensus_out,
            union=union_out,
            image=image_out,
            n_readers=len(present),
        )
        original_mask_dir = dest / "original_masks"
        out = dict(row)
        out.update(aligned_paths)
        out.update(
            {
                "dataset_version": DATASET_VERSION,
                "split": str(row["split"]),
                "n_slices": int(image_out.shape[0]),
                "original_shape_dhw": f"{original_shape[0]}x{original_shape[1]}x{original_shape[2]}",
                "output_shape_dhw": f"{image_out.shape[0]}x{image_out.shape[1]}x{image_out.shape[2]}",
                "images_dir": str(dest / "images"),
                "image_files": "|".join(
                    str(dest / "images" / Path(p).name) for p in image_files
                ),
                "image_volume_path": str(dest / "image_volume.npy"),
                "mask_consensus_path": str(dest / "mask_consensus.npy"),
                "mask_union_path": str(dest / "mask_union.npy"),
                "original_masks_dir": str(original_mask_dir),
                "metadata_path": str(dest / "metadata.json"),
                "intensity_train_low": train_low,
                "intensity_train_high": train_high,
                "intensity_low_percentile": low_p,
                "intensity_high_percentile": high_p,
                "consensus_threshold": consensus_threshold,
                "crop_margin_px": crop_margin_px,
                "crop_y0": y0,
                "crop_y1": y1,
                "crop_x0": x0,
                "crop_x1": x1,
                "pad_offset_y": y_off,
                "pad_offset_x": x_off,
                "foreground_crop_applied": did_crop,
                "n_reader_masks": len(present),
                "consensus_voxels": int(consensus_out.sum()),
                "union_voxels": int(union_out.sum()),
                "image_min": float(image_out.min()),
                "image_max": float(image_out.max()),
                **checks,
            }
        )
        out_rows.append(out)

    result = pd.DataFrame(out_rows)
    if len(result) != EXPECTED_ELIGIBLE_NODULES:
        raise RuntimeError(f"v2 cohort size {len(result)} != {EXPECTED_ELIGIBLE_NODULES}")
    orig_ids = set(manifest["subset_nodule_id"].astype(str))
    new_ids = set(result["subset_nodule_id"].astype(str))
    if orig_ids != new_ids:
        raise RuntimeError("v2 nodule IDs do not match the original 325 cohort.")
    merged_split = manifest[["subset_nodule_id", "split"]].rename(columns={"split": "split_src"})
    check = result.merge(merged_split, on="subset_nodule_id", how="left")
    if not (check["split"].astype(str) == check["split_src"].astype(str)).all():
        raise RuntimeError("Patient-level splits changed while building v2.")
    params["qc_pass_count"] = int(result["qc_pass"].sum())
    params["qc_fail_count"] = int((~result["qc_pass"]).sum())
    return result, params
