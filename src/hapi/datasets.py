"""Slice-level datasets for 2D U-Net consensus segmentation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class LIDCConsensusSegmentationDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, consensus_threshold: float = 0.5):
        self.df = dataframe.reset_index(drop=True).copy()
        self.consensus_threshold = consensus_threshold
        self.index: list[tuple[int, int]] = []
        for row_index, row in self.df.iterrows():
            image_path = str(row["image_volume_path"])
            volume = np.load(image_path, mmap_mode="r")
            for slice_index in range(volume.shape[0]):
                self.index.append((int(row_index), int(slice_index)))

    def __len__(self) -> int:
        return len(self.index)

    def _load_mask(self, row, slice_index: int, mask_number: int):
        column = f"mask_{mask_number}_path"
        if column not in row.index:
            return None
        path = str(row[column])
        if path in {"", "nan"} or path.lower() == "nan":
            return None
        from pathlib import Path

        if not Path(path).exists():
            return None
        mask_volume = np.load(path, mmap_mode="r")
        if slice_index >= mask_volume.shape[0]:
            return None
        mask = np.array(mask_volume[slice_index], dtype=np.float32, copy=True)
        return mask > 0

    def __getitem__(self, index: int) -> dict:
        row_index, slice_index = self.index[index]
        row = self.df.iloc[row_index]
        image_volume = np.load(str(row["image_volume_path"]), mmap_mode="r")
        image = np.clip(
            np.array(image_volume[slice_index], dtype=np.float32, copy=True),
            0.0,
            1.0,
        )

        available = []
        for mask_number in range(4):
            mask = self._load_mask(row, slice_index, mask_number)
            if mask is not None:
                available.append(mask.astype(np.float32))

        if available:
            stacked = np.stack(available, axis=0)
            consensus = (stacked.mean(axis=0) >= self.consensus_threshold).astype(
                np.float32
            )
        else:
            consensus = np.zeros(image.shape, dtype=np.float32)

        return {
            "image": torch.from_numpy(image).unsqueeze(0),
            "mask": torch.from_numpy(consensus).unsqueeze(0),
            "patient_id": str(row["patient_id"]),
            "subset_nodule_id": str(row["subset_nodule_id"]),
            "slice_index": int(slice_index),
        }
