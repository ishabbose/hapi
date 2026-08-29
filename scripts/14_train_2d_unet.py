#!/usr/bin/env python3
"""Step 14 — 2D U-Net on reader-consensus masks (segmentation, not malignancy class)."""

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import common_parser
from hapi.config import CHECKPOINTS_DIR, PREPROCESS_MANIFEST_CSV, SEED, ensure_dirs, require_file
from hapi.datasets import LIDCConsensusSegmentationDataset
from hapi.models import UNet2D
from hapi.train import set_seed


def dice_loss(logits, target, eps=1e-6):
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (probs * target).sum(dim=dims)
    union = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def main() -> None:
    parser = common_parser("Train 2D U-Net consensus segmentation")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    ensure_dirs()
    set_seed(SEED)

    df = pd.read_csv(require_file(PREPROCESS_MANIFEST_CSV, "preprocessed manifest (run 12)"))
    if "split" not in df.columns:
        raise SystemExit("Manifest is missing split. Re-run 10 then 11–12.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = LIDCConsensusSegmentationDataset(df[df["split"] == "train"])
    val_ds = LIDCConsensusSegmentationDataset(df[df["split"] == "val"])
    print(f"Train slices: {len(train_ds)} | Val slices: {len(val_ds)} | Device: {device}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = UNet2D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        n = 0
        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = bce(logits, masks) + dice_loss(logits, masks)
            loss.backward()
            optimizer.step()
            running += loss.item() * images.size(0)
            n += images.size(0)
        model.eval()
        val_loss = 0.0
        vn = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                logits = model(images)
                loss = bce(logits, masks) + dice_loss(logits, masks)
                val_loss += loss.item() * images.size(0)
                vn += images.size(0)
        print(
            f"Epoch {epoch + 1:02d}/{args.epochs} | "
            f"Train: {running / max(n, 1):.4f} | Val: {val_loss / max(vn, 1):.4f}"
        )

    ckpt = CHECKPOINTS_DIR / "unet2d_consensus.pt"
    torch.save(model.state_dict(), ckpt)
    print(f"Saved {ckpt}")


if __name__ == "__main__":
    main()
