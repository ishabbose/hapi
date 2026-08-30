#!/usr/bin/env python3
"""Step 18 — Restrict content/kaggle_dataset_2000.zip to the v2 325-nodule cohort.

Keeps the original unprocessed Kaggle PNGs (and any sidecars) for the same
nodule IDs as data/processed_325_nodules_v2. The first run copies the current
zip to content/kaggle_dataset_2000_original.zip; later runs filter from that
backup so --force stays safe.
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import common_parser
from hapi.config import (
    EXPECTED_ELIGIBLE_NODULES,
    IMPROVED_PROCESSED_DIR,
    KAGGLE_325_ZIP_CSV,
    SUBSET_ZIP,
    SUBSET_ZIP_ORIGINAL,
    ensure_dirs,
    require_file,
)
from hapi.extract import filter_kaggle_zip_to_ids, v2_nodule_ids
from hapi.io_utils import should_skip


def main() -> None:
    args = common_parser(
        "Filter kaggle_dataset_2000.zip to the processed v2 325-nodule IDs"
    ).parse_args()
    ensure_dirs()
    if should_skip(KAGGLE_325_ZIP_CSV, args.force):
        return

    require_file(IMPROVED_PROCESSED_DIR, "processed 325 v2 (run 17)")
    keep = v2_nodule_ids(IMPROVED_PROCESSED_DIR)
    if len(keep) != EXPECTED_ELIGIBLE_NODULES:
        raise SystemExit(
            f"v2 has {len(keep)} nodules, expected {EXPECTED_ELIGIBLE_NODULES}."
        )

    require_file(SUBSET_ZIP, "Kaggle subset ZIP")
    if not SUBSET_ZIP_ORIGINAL.exists():
        print(f"Backing up {SUBSET_ZIP} -> {SUBSET_ZIP_ORIGINAL}")
        shutil.copy2(SUBSET_ZIP, SUBSET_ZIP_ORIGINAL)
    source = SUBSET_ZIP_ORIGINAL
    inventory = filter_kaggle_zip_to_ids(source, SUBSET_ZIP, keep)
    inventory.to_csv(KAGGLE_325_ZIP_CSV, index=False)
    n_kept = int(inventory["kept"].sum())
    print(f"Cohort overlap with v2: {n_kept}/{len(keep)}")
    print(f"Saved {KAGGLE_325_ZIP_CSV}")


if __name__ == "__main__":
    main()
