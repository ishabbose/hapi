"""Run trained autoencoder / U-Net checkpoints on PNG nodule volumes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from hapi.config import CHECKPOINTS_DIR, TARGET_DEPTH, TARGET_SIZE
from hapi.models import SimpleBaseline3DAutoencoder, UNet2D
from hapi.volumes import BaselineCTDataset


def discover_v2_volumes(root_dir: Path) -> dict[str, list[str]]:
    """Use v2 image_volume.npy (train-normalized, crop/padded), not original PNGs."""
    volumes: dict[str, list[str]] = {}
    if not root_dir.exists():
        return volumes
    for npy_path in sorted(root_dir.glob("*/image_volume.npy")):
        volumes[npy_path.parent.name.lower()] = [str(npy_path)]
    return volumes


AE_CHECKPOINT = CHECKPOINTS_DIR / "3d_autoencoder.pt"
UNET_CHECKPOINT = CHECKPOINTS_DIR / "unet2d_consensus.pt"


def load_checkpoint(path: Path, model: torch.nn.Module, device: torch.device) -> None:
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    model.load_state_dict(state)


def consensus_mask_for_slice(case_dir: Path, slice_index: int, threshold: float = 0.5):
    consensus_path = case_dir / "mask_consensus.npy"
    if consensus_path.exists():
        volume = np.load(consensus_path, mmap_mode="r")
        if slice_index < volume.shape[0]:
            return (np.array(volume[slice_index]) > 0).astype(np.float32)
        return None
    masks = []
    for mask_number in range(4):
        npy_path = case_dir / f"mask_{mask_number}.npy"
        if not npy_path.exists():
            continue
        volume = np.load(npy_path, mmap_mode="r")
        if slice_index >= volume.shape[0]:
            continue
        masks.append((np.array(volume[slice_index]) > 0).astype(np.float32))
    if not masks:
        return None
    stacked = np.stack(masks, axis=0)
    return (stacked.mean(axis=0) >= threshold).astype(np.float32)


class IdentifiedVolumeDataset(Dataset):
    def __init__(self, volumes: dict[str, list[str]], volume_ids: list[str]):
        self.ids = volume_ids
        self.inner = BaselineCTDataset(volumes, volume_ids, TARGET_DEPTH, TARGET_SIZE)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        return self.ids[index], self.inner[index]


def _collate_volumes(batch):
    volume_ids = [item[0] for item in batch]
    tensors = torch.stack([item[1] for item in batch], dim=0)
    return volume_ids, tensors


def run_autoencoder(
    volumes: dict[str, list[str]],
    dataset_name: str,
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    if not AE_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Missing autoencoder weights: {AE_CHECKPOINT} (run step 13)"
        )
    model = SimpleBaseline3DAutoencoder().to(device)
    load_checkpoint(AE_CHECKPOINT, model, device)
    model.eval()

    volume_ids = sorted(volumes.keys())
    loader = DataLoader(
        IdentifiedVolumeDataset(volumes, volume_ids),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_volumes,
    )
    rows = []
    criterion = torch.nn.MSELoss(reduction="none")
    with torch.no_grad():
        for ids, batch in loader:
            batch = batch.to(device)
            recon = model(batch)
            per_item = criterion(recon, batch).flatten(1).mean(dim=1)
            for volume_id, mse in zip(ids, per_item.cpu().tolist()):
                files = volumes[volume_id]
                if len(files) == 1 and files[0].lower().endswith(".npy"):
                    n_slices = int(np.load(files[0], mmap_mode="r").shape[0])
                else:
                    n_slices = len(files)
                rows.append(
                    {
                        "dataset": dataset_name,
                        "volume_id": volume_id,
                        "n_slices": n_slices,
                        "mse": float(mse),
                        "checkpoint": str(AE_CHECKPOINT),
                    }
                )
    return pd.DataFrame(rows)


def _load_gray_png(path: str) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("F"), dtype=np.float32)


def _pad_to_multiple(image: torch.Tensor, multiple: int = 8):
    _, _, height, width = image.shape
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    padded = F.pad(image, (0, pad_w, 0, pad_h))
    return padded, height, width


def run_unet(
    volumes: dict[str, list[str]],
    dataset_name: str,
    device: torch.device,
) -> pd.DataFrame:
    if not UNET_CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing U-Net weights: {UNET_CHECKPOINT} (run step 14)")
    model = UNet2D().to(device)
    load_checkpoint(UNET_CHECKPOINT, model, device)
    model.eval()

    rows = []
    with torch.no_grad():
        for volume_id, image_files in sorted(volumes.items()):
            case_dir = Path(image_files[0]).parent
            if case_dir.name.lower() == "images":
                case_dir = case_dir.parent
            if len(image_files) == 1 and image_files[0].lower().endswith(".npy"):
                stacked = np.clip(
                    np.asarray(np.load(image_files[0]), dtype=np.float32), 0.0, 1.0
                )
                if stacked.ndim != 3:
                    raise RuntimeError(
                        f"Expected DHW volume in {image_files[0]}, got {stacked.shape}"
                    )
            else:
                slices = [_load_gray_png(path) for path in image_files]
                stacked = np.stack(slices, axis=0)
                low = float(np.percentile(stacked, 1.0))
                high = float(np.percentile(stacked, 99.0))
                if high <= low:
                    stacked = (stacked - stacked.min()) / (
                        stacked.max() - stacked.min() + 1e-8
                    )
                else:
                    stacked = np.clip(
                        (stacked - low) / (high - low + 1e-8), 0.0, 1.0
                    )

            for slice_index, array in enumerate(stacked):
                image = torch.from_numpy(array).unsqueeze(0).unsqueeze(0).to(device)
                padded, height, width = _pad_to_multiple(image)
                logits = model(padded)[:, :, :height, :width]
                probs = torch.sigmoid(logits)
                pred_bin = (probs >= 0.5).float()
                pred_frac = float(pred_bin.mean().item())

                target = consensus_mask_for_slice(case_dir, slice_index)
                dice = None
                if target is not None:
                    target_t = torch.from_numpy(target).to(device)
                    if target_t.shape[-2:] != (height, width):
                        target_t = F.interpolate(
                            target_t.unsqueeze(0).unsqueeze(0),
                            size=(height, width),
                            mode="nearest",
                        ).squeeze()
                    intersection = (pred_bin.squeeze() * target_t).sum()
                    union = pred_bin.squeeze().sum() + target_t.sum()
                    dice = float(((2 * intersection) / (union + 1e-6)).item())

                rows.append(
                    {
                        "dataset": dataset_name,
                        "volume_id": volume_id,
                        "slice_index": slice_index,
                        "pred_positive_fraction": pred_frac,
                        "dice": dice,
                        "has_consensus_mask": target is not None,
                        "checkpoint": str(UNET_CHECKPOINT),
                    }
                )
    return pd.DataFrame(rows)
