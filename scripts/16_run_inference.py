#!/usr/bin/env python3
"""Step 16 — Run trained checkpoints on processed_325_nodules_v2 and the
filtered content/kaggle_dataset_2000.zip (same 325 unprocessed nodules).
"""

import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import common_parser
from hapi.config import (
    ARTIFACTS_DIR,
    EXPECTED_ELIGIBLE_NODULES,
    IMPROVED_PROCESSED_DIR,
    SUBSET_ZIP,
    ensure_dirs,
)
from hapi.infer import (
    AE_CHECKPOINT,
    UNET_CHECKPOINT,
    discover_v2_volumes,
    run_autoencoder,
    run_unet,
)
from hapi.extract import ensure_kaggle_extracted
from hapi.io_utils import should_skip
from hapi.volumes import discover_volumes


def resolve_datasets(which: str) -> list[tuple[str, Path, str]]:
    """Return (name, root_or_zip, kind) where kind is npy_v2 or png."""
    datasets: list[tuple[str, Path, str]] = []
    v2_volumes = discover_v2_volumes(IMPROVED_PROCESSED_DIR)
    if which in {"processed", "both"}:
        if not v2_volumes:
            raise SystemExit(
                f"No image_volume.npy under {IMPROVED_PROCESSED_DIR}. Run step 17 first."
            )
        datasets.append(("processed_325_nodules_v2", IMPROVED_PROCESSED_DIR, "npy_v2"))
    if which in {"kaggle", "both"}:
        if not SUBSET_ZIP.exists():
            raise SystemExit(f"Missing Kaggle ZIP: {SUBSET_ZIP}")
        extract_dir = ensure_kaggle_extracted(
            zip_path=SUBSET_ZIP, refresh_from_zip=True
        )
        kaggle_volumes = discover_volumes(extract_dir)
        v2_ids = set(v2_volumes) if v2_volumes else set(discover_v2_volumes(IMPROVED_PROCESSED_DIR))
        if not v2_ids:
            raise SystemExit(
                f"No v2 volumes under {IMPROVED_PROCESSED_DIR}. Run step 17 first."
            )
        kaggle_ids = set(kaggle_volumes)
        if kaggle_ids != v2_ids:
            extra = sorted(kaggle_ids - v2_ids)
            missing = sorted(v2_ids - kaggle_ids)
            raise SystemExit(
                "Kaggle ZIP nodule IDs do not match processed_325_nodules_v2. "
                "Run step 18 to filter content/kaggle_dataset_2000.zip to the 325-nodule cohort. "
                f"extra={extra[:10]} missing={missing[:10]} "
                f"(kaggle={len(kaggle_ids)} v2={len(v2_ids)})"
            )
        if len(kaggle_ids) != EXPECTED_ELIGIBLE_NODULES:
            raise SystemExit(
                f"Kaggle ZIP has {len(kaggle_ids)} nodules, expected {EXPECTED_ELIGIBLE_NODULES}."
            )
        datasets.append(("kaggle_dataset_2000", extract_dir, "png"))
    return datasets


def available_models(which: str) -> list[str]:
    models = ["autoencoder", "unet"] if which == "both" else [which]
    ready = []
    for name in models:
        ckpt = AE_CHECKPOINT if name == "autoencoder" else UNET_CHECKPOINT
        if ckpt.exists():
            ready.append(name)
        else:
            print(f"Skipping {name}: missing {ckpt}")
    if not ready:
        raise SystemExit("No checkpoints found. Train with steps 13 and/or 14.")
    return ready


def main() -> None:
    parser = common_parser(
        "Run trained models on processed_325_nodules_v2 and the 325-nodule Kaggle ZIP"
    )
    parser.add_argument(
        "--model",
        choices=["autoencoder", "unet", "both"],
        default="both",
    )
    parser.add_argument(
        "--dataset",
        choices=["processed", "kaggle", "both"],
        default="both",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use only the first N volumes per dataset (smoke test).",
    )
    args = parser.parse_args()
    ensure_dirs()

    ae_out = ARTIFACTS_DIR / "inference_autoencoder.csv"
    unet_out = ARTIFACTS_DIR / "inference_unet.csv"
    models = available_models(args.model)
    if "autoencoder" in models and should_skip(ae_out, args.force):
        models = [m for m in models if m != "autoencoder"]
    if "unet" in models and should_skip(unet_out, args.force):
        models = [m for m in models if m != "unet"]
    if not models:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    datasets = resolve_datasets(args.dataset)

    ae_frames = []
    unet_frames = []
    for dataset_name, root, kind in datasets:
        if kind == "npy_v2":
            volumes = discover_v2_volumes(root)
        else:
            volumes = discover_volumes(root)
        if args.limit is not None:
            keep = sorted(volumes.keys())[: args.limit]
            volumes = {k: volumes[k] for k in keep}
        print(f"{dataset_name}: {len(volumes)} volumes")
        if "autoencoder" in models:
            print(f"  autoencoder …")
            ae_frames.append(
                run_autoencoder(volumes, dataset_name, device, args.batch_size)
            )
        if "unet" in models:
            print(f"  U-Net …")
            unet_frames.append(run_unet(volumes, dataset_name, device))

    if ae_frames:
        ae_df = pd.concat(ae_frames, ignore_index=True)
        ae_df.to_csv(ae_out, index=False)
        print(ae_df.groupby("dataset")["mse"].agg(["count", "mean", "median"]).to_string())
        print(f"Saved {ae_out}")
    if unet_frames:
        unet_df = pd.concat(unet_frames, ignore_index=True)
        unet_df.to_csv(unet_out, index=False)
        print(
            unet_df.groupby("dataset")["pred_positive_fraction"]
            .agg(["count", "mean"])
            .to_string()
        )
        scored = unet_df.dropna(subset=["dice"])
        if len(scored):
            print("Dice (slices with consensus masks):")
            print(scored.groupby("dataset")["dice"].agg(["count", "mean", "median"]).to_string())
        print(f"Saved {unet_out}")


if __name__ == "__main__":
    main()
