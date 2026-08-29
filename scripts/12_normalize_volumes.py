#!/usr/bin/env python3
"""Step 12 — Percentile-normalize crops to [0, 1] numpy volumes."""

import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import common_parser
from hapi.config import (
    MODEL_READY_DIR,
    PREPROCESS_MANIFEST_CSV,
    PROCESSED_MANIFEST_CSV,
    ensure_dirs,
    require_file,
)
from hapi.io_utils import should_skip
from hapi.preprocess import normalize_nodule_volumes


def main() -> None:
    args = common_parser("Build model-ready normalized volumes").parse_args()
    ensure_dirs()
    if should_skip(PREPROCESS_MANIFEST_CSV, args.force):
        return

    df = pd.read_csv(require_file(PROCESSED_MANIFEST_CSV, "processed manifest (run 11)"))
    out = normalize_nodule_volumes(df, MODEL_READY_DIR)
    out.to_csv(PREPROCESS_MANIFEST_CSV, index=False)
    print(f"Saved {PREPROCESS_MANIFEST_CSV} ({len(out)} nodules)")


if __name__ == "__main__":
    main()
