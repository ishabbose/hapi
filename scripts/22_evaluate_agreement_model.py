#!/usr/bin/env python3
"""Evaluate exactly one trained outer-fold checkpoint and save auditable outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hapi.agreement_dataset import (  # noqa: E402
    AgreementVolumeDataset,
    pad_agreement_volumes,
)
from hapi.agreement_metrics import (  # noqa: E402
    SCIPY_AVAILABLE,
    AgreementMetricsAccumulator,
)
from hapi.agreement_model import (  # noqa: E402
    ExactCountResUNet2p5D,
    NestedAgreementResUNet2p5D,
    count_logits_to_cumulative_probabilities,
)
from hapi.models import UNet2D  # noqa: E402


TRAINING_CODE_FILES = (
    REPO_ROOT / "scripts/21_train_agreement_model.py",
    REPO_ROOT / "src/hapi/agreement_dataset.py",
    REPO_ROOT / "src/hapi/agreement_losses.py",
    REPO_ROOT / "src/hapi/agreement_model.py",
    REPO_ROOT / "src/hapi/models.py",
)
EVALUATION_CODE_FILES = (
    REPO_ROOT / "scripts/22_evaluate_agreement_model.py",
    REPO_ROOT / "src/hapi/agreement_dataset.py",
    REPO_ROOT / "src/hapi/agreement_metrics.py",
    REPO_ROOT / "src/hapi/agreement_model.py",
    REPO_ROOT / "src/hapi/models.py",
)


class VolumeUNet2D(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.network = UNet2D(in_channels=in_channels, out_channels=1)

    def forward(
        self,
        image: torch.Tensor,
        depth_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, depth, channels, height, width = image.shape
        logits = self.network(image.reshape(batch * depth, channels, height, width))
        return logits.reshape(batch, depth, 1, height, width).permute(0, 2, 1, 3, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/final_325_nodules_v3")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


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


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return result


def build_model(run_config: dict[str, Any]) -> nn.Module:
    variant = run_config["variant_config"]
    channels = int(variant["context_slices"])
    if variant["model"] == "unet":
        return VolumeUNet2D(channels)
    common = {
        "in_channels": channels,
        "base_channels": int(run_config["base_channels"]),
        "dropout": float(run_config["dropout"]),
        "use_transformer": bool(variant["transformer"]),
    }
    if variant["model"] == "exact_count":
        return ExactCountResUNet2p5D(**common)
    return NestedAgreementResUNet2p5D(
        **common,
        ordered_outputs=bool(variant["ordered_outputs"]),
        output_channels=4 if variant["nested"] else 1,
    )


def cumulative_probabilities(
    logits: torch.Tensor,
    variant: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return four support probabilities and optional exact-count q estimate."""

    if variant["model"] == "exact_count":
        count_probabilities = torch.softmax(logits, dim=1)
        cumulative = count_logits_to_cumulative_probabilities(logits).clamp(0.0, 1.0)
        counts = torch.arange(5, device=logits.device, dtype=logits.dtype)
        q_hat = (
            (count_probabilities * counts[None, :, None, None, None]).sum(dim=1) / 4.0
        ).clamp(0.0, 1.0)
        return cumulative, q_hat
    if not variant["nested"]:
        t2_logits = logits[:, 0] if logits.shape[1] == 1 else logits[:, 1]
        t2_probability = torch.sigmoid(t2_logits)
        # Only T2 is used. Equal placeholder channels let the common metrics
        # implementation reconstruct volumes; non-T2 rows are removed below.
        return t2_probability[:, None].expand(-1, 4, -1, -1, -1), None
    probabilities = torch.sigmoid(logits)
    return probabilities, probabilities.sum(dim=1) / 4.0


def append_metadata(frame: pd.DataFrame, model: str, seed: int, fold: int) -> pd.DataFrame:
    frame = frame.copy()
    frame.insert(0, "model", model)
    frame.insert(1, "seed", seed)
    frame.insert(2, "outer_fold", fold)
    return frame


def calibration_rows(
    q_hat: np.ndarray,
    q: np.ndarray,
    t1: np.ndarray,
    *,
    model: str,
    seed: int,
    outer_fold: int,
    bins: int = 10,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    scopes = {
        "full_roi": np.ones(q.shape, dtype=bool),
        "t1_union": t1.astype(bool),
    }
    for scope, mask in scopes.items():
        predictions = q_hat[mask].astype(np.float64)
        observations = q[mask].astype(np.float64)
        assignments = np.clip(np.digitize(predictions, edges[1:-1], right=False), 0, bins - 1)
        for bin_index in range(bins):
            selected = assignments == bin_index
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "outer_fold": outer_fold,
                    "scope": scope,
                    "bin_lower": float(edges[bin_index]),
                    "bin_upper": float(edges[bin_index + 1]),
                    "mean_predicted": float(predictions[selected].mean()) if selected.any() else np.nan,
                    "mean_observed": float(observations[selected].mean()) if selected.any() else np.nan,
                    "n_voxels": int(selected.sum()),
                }
            )
    return rows


def aggregate_calibration_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Pool per-case reliability bins into one voxel-weighted table per fold."""

    columns = [
        "model",
        "seed",
        "outer_fold",
        "scope",
        "bin_lower",
        "bin_upper",
        "mean_predicted",
        "mean_observed",
        "n_voxels",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows)
    keys = ["model", "seed", "outer_fold", "scope", "bin_lower", "bin_upper"]
    pooled: list[dict[str, Any]] = []
    for identity, group in frame.groupby(keys, sort=True, dropna=False):
        counts = group["n_voxels"].to_numpy(dtype=np.int64)
        total = int(counts.sum())
        row = dict(zip(keys, identity))
        row["n_voxels"] = total
        if total:
            row["mean_predicted"] = float(
                np.nansum(group["mean_predicted"].to_numpy(dtype=np.float64) * counts) / total
            )
            row["mean_observed"] = float(
                np.nansum(group["mean_observed"].to_numpy(dtype=np.float64) * counts) / total
            )
        else:
            row["mean_predicted"] = np.nan
            row["mean_observed"] = np.nan
        pooled.append(row)
    return pd.DataFrame(pooled, columns=columns)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.bootstrap < 1 or args.workers < 0:
        raise ValueError("batch size and bootstrap must be positive; workers must be non-negative")
    if not SCIPY_AVAILABLE:
        raise RuntimeError(
            "SciPy is required for the prespecified HD95 and ASSD endpoints; "
            "install requirements.txt before evaluation"
        )
    run_dir = args.run_dir.resolve()
    data_root = args.data_root.resolve()
    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / "best.pt"
    split_path = run_dir / "split_assignments.csv"
    training_summary_path = run_dir / "run_summary.json"
    output_path = run_dir / "case_metrics.csv"
    summary_path = run_dir / "evaluation_summary.json"
    for required in (config_path, checkpoint_path, split_path, training_summary_path):
        if not required.is_file():
            raise FileNotFoundError(f"Missing training output: {required}")
    if output_path.exists() and not args.force:
        raise FileExistsError(f"{output_path} exists; use --force to reevaluate")
    if args.force:
        # Invalidate the completion sentinel before touching any evaluation
        # artifact.  It is written again only after every output succeeds.
        summary_path.unlink(missing_ok=True)

    run_config = json.loads(config_path.read_text())
    training_summary = json.loads(training_summary_path.read_text())
    evaluation_code_sha256, evaluation_code_files = code_fingerprint(
        EVALUATION_CODE_FILES
    )
    if run_config.get("smoke") and not args.allow_smoke:
        raise RuntimeError("Refusing to report a smoke checkpoint; pass --allow-smoke only for testing")
    model_name = str(run_config["variant"])
    seed = int(run_config["seed"])
    outer_fold = int(run_config["outer_fold"])
    variant = run_config["variant_config"]
    training_code_sha256, _ = code_fingerprint(TRAINING_CODE_FILES)
    if run_config.get("training_code_sha256") != training_code_sha256:
        raise RuntimeError(
            "Training code differs from the checkpoint provenance; retrain this run "
            "with the current locked implementation"
        )
    manifest_path = data_root / "final_325_manifest.csv"
    validation_report = json.loads((data_root / "validation_report.json").read_text())
    if validation_report.get("status") != "PASS":
        raise RuntimeError("Dataset validation does not report PASS")
    if sha256_file(manifest_path) != run_config["manifest_sha256"]:
        raise RuntimeError("Manifest hash differs from the training run")
    if validation_report["dataset_content_sha256"] != run_config["dataset_content_sha256"]:
        raise RuntimeError("Dataset content fingerprint differs from the training run")

    manifest = pd.read_csv(manifest_path)
    outer = manifest[manifest["cv_fold"] == outer_fold].copy()
    outer["split"] = "outer_test"
    assignments = pd.read_csv(split_path)
    expected_ids = set(
        assignments.loc[assignments["experiment_role"] == "outer_test", "subset_nodule_id"].astype(str)
    )
    if set(outer["subset_nodule_id"].astype(str)) != expected_ids:
        raise RuntimeError("Outer-test IDs do not match the frozen training split record")

    normalization = run_config["normalization"]
    dataset = AgreementVolumeDataset(
        outer,
        dataset_root=data_root,
        context_slices=int(variant["context_slices"]),
        intensity_low=float(normalization["intensity_low"]),
        intensity_high=float(normalization["intensity_high"]),
        augment=False,
    )
    device = choose_device(args.device)
    amp_enabled = device.type == "cuda" and not args.no_amp
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=pad_agreement_volumes,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    model = build_model(run_config).to(device)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    expected_training_status = "SMOKE_PASS" if run_config.get("smoke") else "TRAINING_COMPLETE"
    if training_summary.get("status") != expected_training_status:
        raise RuntimeError("Training completion sentinel is absent or inconsistent")
    if training_summary.get("checkpoint_sha256") != checkpoint_sha256:
        raise RuntimeError("Checkpoint hash does not match run_summary.json")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("run_config") != run_config:
        raise RuntimeError("Checkpoint run_config does not exactly match config.json")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    threshold = float(checkpoint["selected_threshold"])
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise RuntimeError("Checkpoint selected_threshold must be finite and in [0, 1]")
    if float(training_summary.get("selected_threshold", float("nan"))) != threshold:
        raise RuntimeError("Checkpoint threshold does not match run_summary.json")
    accumulator = AgreementMetricsAccumulator(
        threshold=threshold,
        expected_split="outer_test",
        empty_policy="skip",
        voxel_spacing=(1.0, 1.0, 1.0),
    )
    predictions_dir = run_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    calibration: list[dict[str, Any]] = []

    with torch.no_grad():
        for raw_batch in loader:
            image = raw_batch["image"].to(device, non_blocking=True)
            depth_valid = raw_batch["depth_valid"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(image, depth_valid)
                probabilities, q_hat = cumulative_probabilities(logits, variant)
            probabilities_cpu = probabilities.float().cpu()
            q_hat_cpu = None if q_hat is None else q_hat.float().cpu()
            accumulator.update_from_volume_batch(
                raw_batch,
                probabilities_cpu,
                from_logits=False,
                q_prediction=q_hat_cpu,
            )

            for index, case_id in enumerate(raw_batch["volume_id"]):
                depth = int(raw_batch["depth"][index])
                case_path = predictions_dir / f"{case_id}.npz"
                if not variant["nested"] and variant["model"] != "exact_count":
                    np.savez_compressed(
                        case_path,
                        t2_probability=probabilities_cpu[index, 1, :depth].numpy().astype(np.float16),
                        threshold=np.asarray(threshold, dtype=np.float32),
                    )
                    continue
                case_probabilities = probabilities_cpu[index, :, :depth].numpy()
                case_q_hat = (
                    q_hat_cpu[index, :depth].numpy()
                    if q_hat_cpu is not None
                    else case_probabilities.mean(axis=0)
                )
                np.savez_compressed(
                    case_path,
                    probabilities=case_probabilities.astype(np.float16),
                    q_hat=case_q_hat.astype(np.float16),
                    threshold=np.asarray(threshold, dtype=np.float32),
                )
                q_target = raw_batch["vote_fraction"][index, :depth].numpy()
                t1_target = raw_batch["targets"][index, 0, :depth].numpy()
                calibration.extend(
                    calibration_rows(
                        case_q_hat,
                        q_target,
                        t1_target,
                        model=model_name,
                        seed=seed,
                        outer_fold=outer_fold,
                    )
                )

    report = accumulator.compute(
        n_bootstrap=args.bootstrap,
        confidence=0.95,
        bootstrap_seed=seed + outer_fold * 1000,
    )
    if not variant["nested"] and variant["model"] != "exact_count":
        case_metrics = report.case_metrics[report.case_metrics["target"] == "T2"].copy()
        patient_metrics = report.patient_metrics[report.patient_metrics["target"] == "T2"].copy()
        all_patient_metrics = report.all_case_patient_metrics[
            report.all_case_patient_metrics["target"] == "T2"
        ].copy()
        target_summary = report.target_summary[report.target_summary["target"] == "T2"].copy()
        bootstrap_cis = report.bootstrap_cis[report.bootstrap_cis["target"] == "T2"].copy()
        soft_case_metrics = pd.DataFrame()
        soft_patient_metrics = pd.DataFrame()
    else:
        case_metrics = report.case_metrics[report.case_metrics["target"].isin(["T1", "T2", "T3", "T4"])].copy()
        patient_metrics = report.patient_metrics[report.patient_metrics["target"].isin(["T1", "T2", "T3", "T4"])].copy()
        all_patient_metrics = report.all_case_patient_metrics[
            report.all_case_patient_metrics["target"].isin(["T1", "T2", "T3", "T4"])
        ].copy()
        target_summary = report.target_summary[report.target_summary["target"].isin(["T1", "T2", "T3", "T4"])].copy()
        bootstrap_cis = report.bootstrap_cis[
            report.bootstrap_cis["target"].isin(["T1", "T2", "T3", "T4", "soft"])
        ].copy()
        soft_case_metrics = report.soft_case_metrics.copy()
        soft_patient_metrics = report.soft_patient_metrics.copy()

    append_metadata(case_metrics, model_name, seed, outer_fold).to_csv(output_path, index=False)
    append_metadata(patient_metrics, model_name, seed, outer_fold).to_csv(
        run_dir / "patient_metrics.csv", index=False
    )
    append_metadata(all_patient_metrics, model_name, seed, outer_fold).to_csv(
        run_dir / "all_case_patient_metrics.csv", index=False
    )
    append_metadata(target_summary, model_name, seed, outer_fold).to_csv(
        run_dir / "target_summary.csv", index=False
    )
    append_metadata(bootstrap_cis, model_name, seed, outer_fold).to_csv(
        run_dir / "bootstrap_cis.csv", index=False
    )
    if not soft_case_metrics.empty:
        append_metadata(soft_case_metrics, model_name, seed, outer_fold).to_csv(
            run_dir / "soft_case_metrics.csv", index=False
        )
        append_metadata(soft_patient_metrics, model_name, seed, outer_fold).to_csv(
            run_dir / "soft_patient_metrics.csv", index=False
        )
        aggregate_calibration_rows(calibration).to_csv(
            run_dir / "calibration.csv", index=False
        )

    t2_summary = target_summary[target_summary["target"] == "T2"].iloc[0].to_dict()
    evaluation_summary = {
        "status": "EVALUATION_COMPLETE",
        "model": model_name,
        "seed": seed,
        "outer_fold": outer_fold,
        "threshold_selected_on_validation": threshold,
        "n_outer_test_cases": int(len(outer)),
        "n_outer_test_patients": int(outer["patient_id"].nunique()),
        "n_t2_positive_cases": int(t2_summary["n_target_positive_cases"]),
        "n_t2_positive_patients": int(t2_summary["n_target_positive_patients"]),
        "patient_macro_t2_dice": float(t2_summary["patient_macro_dice"]),
        "patient_macro_t2_iou": float(t2_summary["patient_macro_iou"]),
        "surface_distance_units": "voxels",
        "bootstrap_replicates": int(args.bootstrap),
        "confidence": 0.95,
        "prediction_directory": str(predictions_dir),
        "checkpoint_sha256": checkpoint_sha256,
        "surface_metrics_available": True,
        "evaluation_code_sha256": evaluation_code_sha256,
        "evaluation_code_files": evaluation_code_files,
    }
    final_evaluation_code_sha256, _ = code_fingerprint(EVALUATION_CODE_FILES)
    if final_evaluation_code_sha256 != evaluation_code_sha256:
        raise RuntimeError("Evaluation code changed while this run was executing; rerun evaluation")
    if sha256_file(checkpoint_path) != checkpoint_sha256:
        raise RuntimeError("Checkpoint changed while this evaluation was executing; rerun evaluation")
    summary_path.write_text(
        json.dumps(evaluation_summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(evaluation_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
