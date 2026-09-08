#!/usr/bin/env python3
"""Train one patient-grouped fold of the HAPI agreement-segmentation study.

Final reporting uses five outer folds.  For outer fold ``f``, fold ``(f+1)%5``
is used only for checkpoint/threshold selection and the remaining three folds
are used for optimization.  The outer fold is never opened by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hapi.agreement_dataset import (  # noqa: E402
    AgreementVolumeDataset,
    pad_agreement_volumes,
)
from hapi.agreement_losses import nested_agreement_loss  # noqa: E402
from hapi.agreement_model import (  # noqa: E402
    ExactCountResUNet2p5D,
    NestedAgreementResUNet2p5D,
    count_logits_to_cumulative_probabilities,
)
from hapi.models import UNet2D  # noqa: E402


VARIANTS: dict[str, dict[str, Any]] = {
    "A0_unet2d_t2": {
        "label": "2D U-Net, hard T2",
        "model": "unet",
        "context_slices": 1,
        "nested": False,
        "ordered_outputs": False,
        "transformer": False,
        "brier_weight": 0.0,
    },
    "A1_resunet2d_t2": {
        "label": "2D residual U-Net, hard T2",
        "model": "resunet",
        "context_slices": 1,
        "nested": False,
        "ordered_outputs": False,
        "transformer": False,
        "brier_weight": 0.0,
    },
    "A2_resunet2p5d_t2": {
        "label": "2.5D residual U-Net, hard T2",
        "model": "resunet",
        "context_slices": 3,
        "nested": False,
        "ordered_outputs": False,
        "transformer": False,
        "brier_weight": 0.0,
    },
    "A3_resunet2d_nested": {
        "label": "2D residual U-Net, ordered T1-T4",
        "model": "resunet",
        "context_slices": 1,
        "nested": True,
        "ordered_outputs": True,
        "transformer": False,
        "brier_weight": 0.10,
    },
    "A4_proposed": {
        "label": "Proposed 2.5D ordered agreement model",
        "model": "resunet",
        "context_slices": 3,
        "nested": True,
        "ordered_outputs": True,
        "transformer": False,
        "brier_weight": 0.10,
    },
    "A5_independent_heads": {
        "label": "2.5D agreement model, independent heads",
        "model": "resunet",
        "context_slices": 3,
        "nested": True,
        "ordered_outputs": False,
        "transformer": False,
        "brier_weight": 0.10,
    },
    "A6_transformer": {
        "label": "2.5D ordered agreement model + transformer",
        "model": "resunet",
        "context_slices": 3,
        "nested": True,
        "ordered_outputs": True,
        "transformer": True,
        "brier_weight": 0.10,
    },
    "A7_ordinal_rps": {
        "label": "Architecture-matched exact-count ORC-RPS",
        "model": "exact_count",
        "context_slices": 3,
        "nested": False,
        "ordinal_rps": True,
        "ordered_outputs": True,
        "transformer": False,
        "brier_weight": 0.0,
    },
    "A8_no_brier": {
        "label": "Proposed 2.5D ordered model without Brier term",
        "model": "resunet",
        "context_slices": 3,
        "nested": True,
        "ordered_outputs": True,
        "transformer": False,
        "brier_weight": 0.0,
    },
}

TRAINING_CODE_FILES = (
    REPO_ROOT / "scripts/21_train_agreement_model.py",
    REPO_ROOT / "src/hapi/agreement_dataset.py",
    REPO_ROOT / "src/hapi/agreement_losses.py",
    REPO_ROOT / "src/hapi/agreement_model.py",
    REPO_ROOT / "src/hapi/models.py",
)


class VolumeUNet2D(nn.Module):
    """Apply the repository's 2D U-Net independently to padded volume slices."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.network = UNet2D(in_channels=in_channels, out_channels=1)

    def forward(
        self,
        image: torch.Tensor,
        depth_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if image.ndim != 5:
            raise ValueError("VolumeUNet2D expects [B,D,C,H,W]")
        batch, depth, channels, height, width = image.shape
        logits = self.network(image.reshape(batch * depth, channels, height, width))
        logits = logits.reshape(batch, depth, 1, height, width)
        return logits.permute(0, 2, 1, 3, 4).contiguous()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="A4_proposed")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/final_325_nodules_v3")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "results/final_agreement")
    parser.add_argument("--outer-fold", type=int, choices=range(5), default=0)
    parser.add_argument("--validation-fold", type=int, choices=range(5), default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a two-train/two-validation-case integration check in a separate output tree.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_fingerprint(paths: tuple[Path, ...]) -> tuple[str, dict[str, str]]:
    digest = hashlib.sha256()
    file_hashes: dict[str, str] = {}
    for path in paths:
        relative = str(path.resolve().relative_to(REPO_ROOT))
        value = sha256_file(path)
        file_hashes[relative] = value
        digest.update(relative.encode("utf-8") + b"\0" + value.encode("ascii") + b"\n")
    return digest.hexdigest(), file_hashes


def percentile_from_histogram(histogram: np.ndarray, percentile: float) -> float:
    cumulative = np.cumsum(histogram)
    target = percentile / 100.0 * cumulative[-1]
    return float(np.searchsorted(cumulative, target, side="left"))


def fit_intensity_window(
    frame: pd.DataFrame,
    data_root: Path,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
) -> tuple[float, float]:
    """Fit a raw-PNG intensity window on optimization patients only."""

    histogram = np.zeros(256, dtype=np.int64)
    for relative_path in frame["image_volume_uint8_path"]:
        volume = np.load(data_root / str(relative_path), mmap_mode="r")
        if volume.dtype != np.uint8:
            raise TypeError(f"Expected uint8 image volume: {relative_path}")
        histogram += np.bincount(np.asarray(volume).ravel(), minlength=256)
    low = percentile_from_histogram(histogram, low_percentile)
    high = percentile_from_histogram(histogram, high_percentile)
    if high <= low:
        raise ValueError(f"Invalid optimization-fold intensity window: {low}, {high}")
    return low, high


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def make_grad_scaler(enabled: bool):
    """AMP scaler compatible with PyTorch 2.2 (cuda.amp) and 2.4+ (torch.amp)."""
    amp_module = getattr(torch, "amp", None)
    if amp_module is not None and hasattr(amp_module, "GradScaler"):
        return amp_module.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def prepare_folds(
    manifest: pd.DataFrame,
    outer_fold: int,
    validation_fold: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    validation_fold = (outer_fold + 1) % 5 if validation_fold is None else validation_fold
    if validation_fold == outer_fold:
        raise ValueError("Validation fold must differ from the outer test fold")
    if set(manifest["cv_fold"].astype(int)) != set(range(5)):
        raise ValueError("Manifest must contain outer folds 0 through 4")
    if int(manifest.groupby("patient_id")["cv_fold"].nunique().max()) != 1:
        raise ValueError("Patient leakage exists across cv_fold values")

    test = manifest[manifest["cv_fold"] == outer_fold].copy()
    validation = manifest[manifest["cv_fold"] == validation_fold].copy()
    train = manifest[~manifest["cv_fold"].isin([outer_fold, validation_fold])].copy()
    train["split"] = "outer_train"
    validation["split"] = "inner_val"
    test["split"] = "outer_test"

    patient_sets = [set(frame["patient_id"].astype(str)) for frame in (train, validation, test)]
    if patient_sets[0] & patient_sets[1] or patient_sets[0] & patient_sets[2] or patient_sets[1] & patient_sets[2]:
        raise ValueError("Patient overlap detected among optimization, validation, and outer test roles")
    return train, validation, test, validation_fold


def build_model(config: dict[str, Any], args: argparse.Namespace) -> nn.Module:
    channels = int(config["context_slices"])
    if config["model"] == "unet":
        return VolumeUNet2D(channels)
    if config["model"] == "exact_count":
        return ExactCountResUNet2p5D(
            in_channels=channels,
            base_channels=args.base_channels,
            dropout=args.dropout,
            use_transformer=bool(config["transformer"]),
        )
    return NestedAgreementResUNet2p5D(
        in_channels=channels,
        base_channels=args.base_channels,
        dropout=args.dropout,
        use_transformer=bool(config["transformer"]),
        ordered_outputs=bool(config["ordered_outputs"]),
        output_channels=4 if config["nested"] else 1,
    )


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    for key in ("image", "targets", "valid_heads", "vote_fraction", "majority", "depth_valid"):
        moved[key] = batch[key].to(device, non_blocking=True)
    return moved


def focal_dice_t2_loss(
    logits: torch.Tensor,
    batch: dict[str, Any],
    *,
    alpha: float = 0.75,
    gamma: float = 2.0,
    epsilon: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Case-balanced focal BCE on all cases plus 3D Dice on T2-positive cases."""

    t2_logits = logits[:, 0:1] if logits.shape[1] == 1 else logits[:, 1:2]
    target = batch["targets"][:, 1:2].to(t2_logits.dtype)
    included = batch["valid_heads"][:, 1]
    depth_mask = batch["depth_valid"][:, None, :, None, None].to(t2_logits.dtype)
    probabilities = torch.sigmoid(t2_logits)

    cross_entropy = F.binary_cross_entropy_with_logits(t2_logits, target, reduction="none")
    p_target = probabilities * target + (1.0 - probabilities) * (1.0 - target)
    alpha_target = alpha * target + (1.0 - alpha) * (1.0 - target)
    focal_voxel = alpha_target * (1.0 - p_target).pow(gamma) * cross_entropy
    valid_voxels = batch["depth_valid"].sum(dim=1).to(t2_logits.dtype) * float(
        target.shape[-2] * target.shape[-1]
    )
    focal_per_case = (focal_voxel * depth_mask).sum(dim=(1, 2, 3, 4)) / valid_voxels

    masked_probability = probabilities * depth_mask
    masked_target = target * depth_mask
    intersection = (masked_probability * masked_target).sum(dim=(1, 2, 3, 4))
    denominator = masked_probability.sum(dim=(1, 2, 3, 4)) + masked_target.sum(
        dim=(1, 2, 3, 4)
    )
    dice_per_case = 1.0 - (2.0 * intersection + epsilon) / (denominator + epsilon)
    focal_weights = included.to(t2_logits.dtype)
    focal_normalizer = focal_weights.sum().clamp_min(1.0)
    focal = (focal_per_case * focal_weights).sum() / focal_normalizer
    target_positive = target.sum(dim=(1, 2, 3, 4)) > 0
    dice_included = included & target_positive
    dice_weights = dice_included.to(t2_logits.dtype)
    dice_normalizer = dice_weights.sum().clamp_min(1.0)
    dice = (dice_per_case * dice_weights).sum() / dice_normalizer
    if not bool(dice_included.any().item()):
        dice = t2_logits.sum() * 0.0
    loss = 0.5 * focal + 0.5 * dice
    # Preserve a gradient-bearing zero for a theoretically all-invalid batch.
    if not bool(included.any().item()):
        loss = t2_logits.sum() * 0.0
        focal = loss
        dice = loss
    return {"loss": loss, "focal": focal, "dice": dice, "brier": loss * 0.0}


def ordinal_rps_loss(
    count_logits: torch.Tensor,
    batch: dict[str, Any],
    *,
    alpha: float = 0.8,
) -> dict[str, torch.Tensor]:
    """Architecture-matched BCE + ranked probability score comparator.

    This implements Riera-Marín et al.'s K+1 exact-count formulation for K=4
    while retaining the same 2.5D residual backbone used by the proposed model.
    """

    if count_logits.ndim != 5 or count_logits.shape[1] != 5:
        raise ValueError("ORC-RPS logits must have shape [B,5,D,H,W]")
    target_count = batch["targets"].sum(dim=1).round().long()
    probabilities = torch.softmax(count_logits, dim=1)
    # Express P(C>=2) versus P(C<2) as a single stable logit.  BCE on a
    # softmax probability is unsafe under CUDA autocast and can round to an
    # exact endpoint in float16; the grouped log-sum-exp form is equivalent.
    foreground_logit = torch.logsumexp(count_logits[:, 2:], dim=1) - torch.logsumexp(
        count_logits[:, :2], dim=1
    )
    foreground_target = (target_count >= 2).to(count_logits.dtype)
    valid = batch["depth_valid"][:, :, None, None].to(count_logits.dtype)
    voxels_per_case = (
        valid.sum(dim=(1, 2, 3)) * float(target_count.shape[-2] * target_count.shape[-1])
    ).clamp_min(1.0)

    bce_voxel = F.binary_cross_entropy_with_logits(
        foreground_logit,
        foreground_target,
        reduction="none",
    )
    bce_per_case = (bce_voxel * valid).sum(dim=(1, 2, 3)) / voxels_per_case

    # Match Riera-Marín et al. Eq. (3): 1/(K+1) over j=0,...,K.  The terminal
    # CDF entry is identically one but is retained to reproduce their stated
    # five-class loss scaling exactly.
    predicted_cdf = probabilities.cumsum(dim=1)
    classes = torch.arange(5, device=count_logits.device).view(1, 5, 1, 1, 1)
    observed_cdf = (target_count[:, None] <= classes).to(count_logits.dtype)
    rps_voxel = (predicted_cdf - observed_cdf).pow(2).mean(dim=1)
    rps_per_case = (rps_voxel * valid).sum(dim=(1, 2, 3)) / voxels_per_case
    bce = bce_per_case.mean()
    rps = rps_per_case.mean()
    loss = bce + alpha * rps
    return {"loss": loss, "focal": bce, "dice": rps, "brier": loss * 0.0}


def compute_loss(
    logits: torch.Tensor,
    batch: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    if config.get("ordinal_rps", False):
        return ordinal_rps_loss(logits, batch)
    if not config["nested"]:
        return focal_dice_t2_loss(logits, batch)
    return nested_agreement_loss(
        logits,
        batch["targets"],
        batch["valid_heads"],
        batch["vote_fraction"],
        batch["depth_valid"],
        brier_weight=float(config["brier_weight"]),
    )


def t2_probabilities(logits: torch.Tensor) -> torch.Tensor:
    if logits.shape[1] == 5:
        return count_logits_to_cumulative_probabilities(logits)[:, 1]
    channel = 0 if logits.shape[1] == 1 else 1
    return torch.sigmoid(logits[:, channel])


def batch_t2_records(
    logits: torch.Tensor,
    batch: dict[str, Any],
) -> list[dict[str, Any]]:
    probabilities = t2_probabilities(logits).detach().cpu().numpy()
    targets = batch["targets"][:, 1].detach().cpu().numpy().astype(bool)
    valid_heads = batch["valid_heads"].detach().cpu().numpy().astype(bool)
    depth_valid = batch["depth_valid"].detach().cpu().numpy().astype(bool)
    records = []
    for index, is_valid in enumerate(valid_heads[:, 1]):
        if not is_valid:
            continue
        depth = int(depth_valid[index].sum())
        target_volume = targets[index, :depth].copy()
        # The prespecified overlap endpoint uses target-positive T2 cases;
        # empty–empty cases are evaluated separately as support-presence events.
        if not bool(target_volume.any()):
            continue
        records.append(
            {
                "volume_id": str(batch["volume_id"][index]),
                "patient_id": str(batch["patient_id"][index]),
                "probability": probabilities[index, :depth].copy(),
                "target": target_volume,
            }
        )
    return records


def patient_macro_dice(records: list[dict[str, Any]], threshold: float) -> float:
    patient_values: dict[str, list[float]] = defaultdict(list)
    for record in records:
        prediction = record["probability"] >= threshold
        target = record["target"]
        denominator = int(prediction.sum()) + int(target.sum())
        dice = 1.0 if denominator == 0 else 2.0 * int(np.logical_and(prediction, target).sum()) / denominator
        patient_values[record["patient_id"]].append(float(dice))
    if not patient_values:
        return float("nan")
    return float(np.mean([np.mean(values) for values in patient_values.values()]))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    amp_enabled: bool,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    component_sums = defaultdict(float)
    n_batches = 0
    records: list[dict[str, Any]] = []
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(batch["image"], batch["depth_valid"])
            parts = compute_loss(logits, batch, config)
        for key in ("loss", "focal", "dice", "brier"):
            component_sums[key] += float(parts[key].detach().cpu())
        n_batches += 1
        records.extend(batch_t2_records(logits, batch))
    metrics = {f"val_{key}": value / max(n_batches, 1) for key, value in component_sums.items()}
    metrics["val_patient_macro_t2_dice"] = patient_macro_dice(records, 0.5)
    return metrics, records


def worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.accumulation_steps < 1:
        raise ValueError("epochs, patience, batch size, and accumulation steps must be positive")
    set_seed(args.seed)
    device = choose_device(args.device)
    config = dict(VARIANTS[args.variant])

    data_root = args.data_root.resolve()
    manifest_path = data_root / "final_325_manifest.csv"
    validation_path = data_root / "validation_report.json"
    if not manifest_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError("Run scripts/19_build_final_v3.py and 20_validate_final_v3.py first")
    validation_report = json.loads(validation_path.read_text())
    if validation_report.get("status") != "PASS":
        raise RuntimeError("Dataset validation_report.json does not report PASS")
    manifest = pd.read_csv(manifest_path)
    train_frame, val_frame, test_frame, validation_fold = prepare_folds(
        manifest, args.outer_fold, args.validation_fold
    )
    normalization_frame = train_frame.copy()
    intensity_low, intensity_high = fit_intensity_window(normalization_frame, data_root)

    # Every case has four review-session slots.  An empty T2 mask is therefore
    # a valid negative target, not a missing label; all cases remain in every
    # matched training and validation cohort.

    run_root = args.output_root.resolve()
    if args.smoke:
        run_root = run_root / "_smoke"
        train_frame = train_frame.head(2).copy()
        val_frame = val_frame.head(2).copy()
        args.epochs = 1
        args.patience = 1
        args.base_channels = min(args.base_channels, 4)
        args.accumulation_steps = 1
    run_dir = run_root / args.variant / f"seed_{args.seed}" / f"fold_{args.outer_fold}"
    checkpoint_path = run_dir / "best.pt"
    if checkpoint_path.exists() and not args.force:
        raise FileExistsError(f"{checkpoint_path} exists; use --force or choose a new output root")
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.force:
        # These files are completion sentinels.  Remove them before replacing
        # the configuration so an interrupted restart cannot be mistaken for
        # a completed training/evaluation run.
        for stale_name in ("run_summary.json", "evaluation_summary.json", "best.pt"):
            (run_dir / stale_name).unlink(missing_ok=True)

    train_dataset = AgreementVolumeDataset(
        train_frame,
        dataset_root=data_root,
        context_slices=int(config["context_slices"]),
        intensity_low=intensity_low,
        intensity_high=intensity_high,
        augment=True,
        augmentation_seed=args.seed,
    )
    val_dataset = AgreementVolumeDataset(
        val_frame,
        dataset_root=data_root,
        context_slices=int(config["context_slices"]),
        intensity_low=intensity_low,
        intensity_high=intensity_high,
        augment=False,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "collate_fn": pad_agreement_volumes,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": worker_seed,
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

    model = build_model(config, args).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    amp_enabled = device.type == "cuda" and not args.no_amp
    scaler = make_grad_scaler(amp_enabled)

    split_assignments = pd.concat(
        [
            train_frame.assign(experiment_role="optimization"),
            val_frame.assign(experiment_role="validation"),
            test_frame.assign(experiment_role="outer_test"),
        ],
        ignore_index=True,
    )[["subset_nodule_id", "patient_id", "cv_fold", "experiment_role", "n_clustered_contours"]]
    split_assignments.to_csv(run_dir / "split_assignments.csv", index=False)
    training_code_sha256, training_code_files = code_fingerprint(TRAINING_CODE_FILES)
    run_config = {
        "variant": args.variant,
        "variant_config": config,
        "outer_fold": args.outer_fold,
        "validation_fold": validation_fold,
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "accumulation_steps": args.accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "base_channels": args.base_channels,
        "dropout": args.dropout,
        "device": str(device),
        "amp": amp_enabled,
        "torch_version": torch.__version__,
        "training_code_sha256": training_code_sha256,
        "training_code_files": training_code_files,
        "manifest_sha256": sha256_file(manifest_path),
        "dataset_content_sha256": validation_report["dataset_content_sha256"],
        "normalization": {
            "fit_role": "optimization",
            "low_percentile": 1.0,
            "high_percentile": 99.0,
            "intensity_low": intensity_low,
            "intensity_high": intensity_high,
        },
        "n_train_cases": len(train_frame),
        "n_validation_cases": len(val_frame),
        "n_outer_test_cases_unopened": len(test_frame),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "smoke": args.smoke,
    }
    (run_dir / "config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n")

    history: list[dict[str, float | int]] = []
    best_score = -np.inf
    best_loss = np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch - 1)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sums = defaultdict(float)
        n_batches = 0
        for batch_index, raw_batch in enumerate(train_loader, start=1):
            batch = move_batch(raw_batch, device)
            group_start = ((batch_index - 1) // args.accumulation_steps) * args.accumulation_steps + 1
            group_size = min(
                args.accumulation_steps,
                len(train_loader) - group_start + 1,
            )
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(batch["image"], batch["depth_valid"])
                parts = compute_loss(logits, batch, config)
                scaled_loss = parts["loss"] / group_size
            scaler.scale(scaled_loss).backward()
            should_step = batch_index % args.accumulation_steps == 0 or batch_index == len(train_loader)
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            for key in ("loss", "focal", "dice", "brier"):
                sums[key] += float(parts[key].detach().cpu())
            n_batches += 1
        scheduler.step()

        val_metrics, _ = evaluate(model, val_loader, device, config, amp_enabled)
        row: dict[str, float | int] = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **{f"train_{key}": value / max(n_batches, 1) for key, value in sums.items()},
            **val_metrics,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

        score = float(val_metrics["val_patient_macro_t2_dice"])
        val_loss = float(val_metrics["val_loss"])
        improved = score > best_score + 1e-6 or (
            abs(score - best_score) <= 1e-6 and val_loss < best_loss
        )
        if improved:
            best_score = score
            best_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "format_version": 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_validation_patient_macro_t2_dice": score,
                    "best_validation_loss": val_loss,
                    "selected_threshold": 0.5,
                    "run_config": run_config,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        print(
            f"{args.variant} fold={args.outer_fold} seed={args.seed} "
            f"epoch={epoch:03d} train={row['train_loss']:.4f} "
            f"val={val_loss:.4f} val_T2={score:.4f}"
        )
        if epochs_without_improvement >= args.patience:
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, validation_records = evaluate(model, val_loader, device, config, amp_enabled)
    threshold_rows = []
    for threshold in np.round(np.arange(0.15, 0.851, 0.025), 3):
        threshold_rows.append(
            {
                "threshold": float(threshold),
                "patient_macro_t2_dice": patient_macro_dice(validation_records, float(threshold)),
            }
        )
    threshold_frame = pd.DataFrame(threshold_rows)
    threshold_frame.to_csv(run_dir / "validation_thresholds.csv", index=False)
    selected_threshold = float(
        threshold_frame.sort_values(
            ["patient_macro_t2_dice", "threshold"], ascending=[False, True]
        ).iloc[0]["threshold"]
    )
    checkpoint["selected_threshold"] = selected_threshold
    torch.save(checkpoint, checkpoint_path)
    final_training_code_sha256, _ = code_fingerprint(TRAINING_CODE_FILES)
    if final_training_code_sha256 != training_code_sha256:
        checkpoint_path.unlink(missing_ok=True)
        raise RuntimeError("Training code changed while this run was executing; restart the run")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    summary = {
        "status": "SMOKE_PASS" if args.smoke else "TRAINING_COMPLETE",
        "best_epoch": best_epoch,
        "best_validation_patient_macro_t2_dice_at_0_5": best_score,
        "best_validation_loss": best_loss,
        "selected_threshold": selected_threshold,
        "selected_threshold_validation_patient_macro_t2_dice": float(
            threshold_frame[threshold_frame["threshold"] == selected_threshold][
                "patient_macro_t2_dice"
            ].iloc[0]
        ),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
