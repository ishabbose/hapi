#!/usr/bin/env python3
"""Step 13 — Unsupervised 3D autoencoder reconstruction baseline (no cancer labels).

Uses extracted subset PNGs if available; otherwise uses processed_325_nodules/images.
"""

import zipfile

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import common_parser
from hapi.config import (
    CHECKPOINTS_DIR,
    PROCESSED_DIR,
    SEED,
    SUBSET_EXTRACT_DIR,
    SUBSET_ZIP,
    TARGET_DEPTH,
    TARGET_SIZE,
    ensure_dirs,
)
from hapi.train import train_autoencoder
from hapi.volumes import BaselineCTDataset, discover_volumes, volume_id_split


def maybe_extract_subset() -> None:
    if SUBSET_EXTRACT_DIR.exists() and any(SUBSET_EXTRACT_DIR.rglob("*.png")):
        return
    if not SUBSET_ZIP.exists():
        return
    print(f"Extracting subset ZIP to {SUBSET_EXTRACT_DIR}")
    SUBSET_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(SUBSET_ZIP, "r") as z:
        z.extractall(SUBSET_EXTRACT_DIR)


def main() -> None:
    parser = common_parser("Train 3D convolutional autoencoder")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    ensure_dirs()

    maybe_extract_subset()
    if SUBSET_EXTRACT_DIR.exists() and any(SUBSET_EXTRACT_DIR.rglob("*.png")):
        root = SUBSET_EXTRACT_DIR
    elif PROCESSED_DIR.exists() and any(PROCESSED_DIR.rglob("*.png")):
        root = PROCESSED_DIR
    else:
        raise SystemExit(
            "No PNG volumes found. Place kaggle_dataset_2000.zip in content/ "
            "or run scripts 11 first."
        )

    volumes = discover_volumes(root)
    print(f"Discovered {len(volumes)} volumes under {root}")
    train_ids, val_ids, test_ids = volume_id_split(list(volumes.keys()), seed=SEED)
    print(f"Split volumes train/val/test: {len(train_ids)}/{len(val_ids)}/{len(test_ids)}")

    train_ds = BaselineCTDataset(volumes, train_ids, TARGET_DEPTH, TARGET_SIZE)
    val_ds = BaselineCTDataset(volumes, val_ids, TARGET_DEPTH, TARGET_SIZE)
    test_ds = BaselineCTDataset(volumes, test_ids, TARGET_DEPTH, TARGET_SIZE)

    ckpt = CHECKPOINTS_DIR / "3d_autoencoder.pt"
    metrics = train_autoencoder(
        train_ds,
        val_ds,
        test_ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=SEED,
        checkpoint_path=ckpt,
    )
    print(metrics)


if __name__ == "__main__":
    main()
