"""Volume discovery, patient splits, and 3D crop loading."""

from __future__ import annotations

import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def natural_sort_key(filename: str):
    numbers = re.findall(r"\d+", Path(filename).name)
    return int(numbers[-1]) if numbers else filename


def discover_volumes(root_dir: Path) -> dict[str, list[str]]:
    volumes: dict[str, list[str]] = {}
    for current_root, dirs, files_in_dir in os_walk_safe(root_dir):
        png_files = [
            f
            for f in files_in_dir
            if f.lower().endswith(".png") and not f.startswith(".")
        ]
        if not png_files:
            continue
        folder_name = Path(current_root).name
        nodule_match = re.search(r"(nodule_\d+)", folder_name, re.IGNORECASE)
        vol_id = nodule_match.group(1).lower() if nodule_match else folder_name.lower()
        if vol_id in volumes:
            raise RuntimeError(
                f"Duplicate volume ID '{vol_id}' at {current_root} and {volumes[vol_id][0]}"
            )
        png_files.sort(key=natural_sort_key)
        volumes[vol_id] = [str(Path(current_root) / f) for f in png_files]
    return volumes


def os_walk_safe(root_dir: Path):
    import os

    for current_root, dirs, files_in_dir in os.walk(root_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__MACOSX"]
        yield current_root, dirs, files_in_dir


def patient_isolated_split(
    df: pd.DataFrame,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> pd.DataFrame:
    patients = sorted(df["patient_id"].astype(str).unique())
    rng = random.Random(seed)
    rng.shuffle(patients)

    n_patients = len(patients)
    n_train = int(round(train_frac * n_patients))
    n_val = int(round(val_frac * n_patients))
    train_patients = set(patients[:n_train])
    val_patients = set(patients[n_train : n_train + n_val])
    test_patients = set(patients[n_train + n_val :])

    def assign_split(patient_id: str) -> str:
        patient_id = str(patient_id)
        if patient_id in train_patients:
            return "train"
        if patient_id in val_patients:
            return "val"
        if patient_id in test_patients:
            return "test"
        raise ValueError(f"Unassigned patient: {patient_id}")

    out = df.copy()
    out["split"] = out["patient_id"].map(assign_split)

    train_ids = set(out.loc[out["split"] == "train", "patient_id"])
    val_ids = set(out.loc[out["split"] == "val", "patient_id"])
    test_ids = set(out.loc[out["split"] == "test", "patient_id"])
    assert not (train_ids & val_ids)
    assert not (train_ids & test_ids)
    assert not (val_ids & test_ids)
    return out


def volume_id_split(
    volume_ids: list[str],
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    shuffled = np.array(sorted(volume_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(shuffled)
    total = len(shuffled)
    train_end = int(0.70 * total)
    val_end = int(0.85 * total)
    return (
        shuffled[:train_end].tolist(),
        shuffled[train_end:val_end].tolist(),
        shuffled[val_end:].tolist(),
    )


class BaselineCTDataset(Dataset):
    """Resize each nodule crop to (1, D, H, W) and min-max normalize."""

    def __init__(
        self,
        volume_dict: dict[str, list[str]],
        volume_ids: list[str],
        target_depth: int = 16,
        target_size: tuple[int, int] = (64, 64),
    ):
        self.volume_dict = volume_dict
        self.volume_ids = volume_ids
        self.target_depth = target_depth
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.volume_ids)

    def __getitem__(self, idx: int) -> torch.Tensor:
        from skimage import io

        vol_id = self.volume_ids[idx]
        image_files = self.volume_dict[vol_id]
        if not image_files:
            raise RuntimeError(f"Volume '{vol_id}' has zero slices.")

        slices = []
        for img_path in image_files:
            img = io.imread(img_path, as_gray=True)
            img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            img_resized = nn.functional.interpolate(
                img_tensor,
                size=self.target_size,
                mode="bilinear",
                align_corners=False,
            )
            slices.append(img_resized.squeeze(0).squeeze(0))

        volume = torch.stack(slices, dim=0).unsqueeze(0)
        volume = nn.functional.interpolate(
            volume.unsqueeze(0),
            size=(self.target_depth, self.target_size[0], self.target_size[1]),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
        return volume
