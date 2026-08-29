"""Percentile windowing and numpy volume export for model-ready arrays."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def load_png(path: str) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("F"), dtype=np.float32)


def percentile_normalize(
    volume: np.ndarray,
    low_p: float = 1.0,
    high_p: float = 99.0,
) -> np.ndarray:
    low = float(np.percentile(volume, low_p))
    high = float(np.percentile(volume, high_p))
    if high <= low:
        denom = float(volume.max() - volume.min()) + 1e-8
        return (volume - float(volume.min())) / denom
    return np.clip((volume - low) / (high - low + 1e-8), 0.0, 1.0)


def normalize_nodule_volumes(
    nodule_manifest: pd.DataFrame,
    output_root: Path,
    image_col: str = "image_files",
) -> pd.DataFrame:
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []

    for _, row in nodule_manifest.iterrows():
        subset_id = str(row["subset_nodule_id"])
        output_dir = output_root / subset_id
        output_dir.mkdir(parents=True, exist_ok=True)

        if image_col in row and pd.notna(row[image_col]):
            image_files = [x for x in str(row[image_col]).split("|") if x]
        else:
            images_dir = Path(str(row.get("images_dir", "")))
            image_files = sorted(str(p) for p in images_dir.glob("*.png"))

        if not image_files:
            print(f"WARNING: no images for {subset_id}")
            continue

        arrays = [load_png(path) for path in image_files]
        image_volume = np.stack(arrays, axis=0)
        normalized = percentile_normalize(image_volume)
        volume_path = output_dir / "image_volume.npy"
        np.save(volume_path, normalized.astype(np.float32))

        out = dict(row)
        out["image_volume_path"] = str(volume_path)
        out["n_slices"] = int(normalized.shape[0])
        rows.append(out)

    return pd.DataFrame(rows)
