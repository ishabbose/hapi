#!/usr/bin/env python3
"""Step 15 — Attach TCIA .tcia basket metadata to processed 325 and Kaggle scans.

The manifest itself only has download-basket headers plus SeriesInstanceUIDs.
Matching scans get those fields plus the NBIA series row for the same UID
(from step 06).

Processed 325: updates each case's metadata.json (same as before).
Kaggle 2000: writes metadata.json next to each nodule's PNGs in the extract
and adds those files into content/kaggle_dataset_2000.zip.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import common_parser
from hapi.config import (
    KAGGLE_TCIA_METADATA_CSV,
    PROCESSED_DIR,
    PROCESSED_TCIA_METADATA_CSV,
    SUBSET_ZIP,
    TCIA_MANIFEST,
    TCIA_SERIES_CSV,
    ensure_dirs,
    require_file,
)
from hapi.extract import ensure_kaggle_extracted
from hapi.io_utils import should_skip
from hapi.tcia import (
    attach_tcia_metadata_to_kaggle,
    attach_tcia_metadata_to_processed,
)


def _series_csv():
    if TCIA_SERIES_CSV.exists():
        return TCIA_SERIES_CSV
    print(
        f"WARNING: {TCIA_SERIES_CSV} not found. "
        "Attaching basket headers only. Run step 06 for per-series fields."
    )
    return None


def _print_summary(label: str, attached, csv_path) -> None:
    n = len(attached)
    n_in = int(attached["in_tcia_manifest"].sum()) if n else 0
    n_series = int(attached["has_tcia_series_metadata"].sum()) if n else 0
    print(f"{label}: updated metadata.json for {n} nodules")
    print(f"  Series UID listed in .tcia: {n_in}/{n}")
    print(f"  With NBIA series metadata: {n_series}/{n}")
    print(f"  Saved {csv_path}")


def attach_processed(force: bool, series_csv) -> None:
    if should_skip(PROCESSED_TCIA_METADATA_CSV, force):
        return
    require_file(PROCESSED_DIR, "processed 325 nodules (run 11)")
    if not any(PROCESSED_DIR.iterdir()):
        raise SystemExit(f"No cases under {PROCESSED_DIR}. Run step 11 first.")
    attached = attach_tcia_metadata_to_processed(
        PROCESSED_DIR,
        TCIA_MANIFEST,
        series_csv=series_csv,
    )
    attached.to_csv(PROCESSED_TCIA_METADATA_CSV, index=False)
    _print_summary("processed_325_nodules", attached, PROCESSED_TCIA_METADATA_CSV)


def attach_kaggle(force: bool, series_csv) -> None:
    if should_skip(KAGGLE_TCIA_METADATA_CSV, force):
        return
    require_file(SUBSET_ZIP, "Kaggle subset ZIP")
    extract_dir = ensure_kaggle_extracted()
    attached = attach_tcia_metadata_to_kaggle(
        extract_dir,
        SUBSET_ZIP,
        TCIA_MANIFEST,
        series_csv=series_csv,
    )
    attached.to_csv(KAGGLE_TCIA_METADATA_CSV, index=False)
    _print_summary("kaggle_dataset_2000", attached, KAGGLE_TCIA_METADATA_CSV)


def main() -> None:
    args = common_parser(
        "Attach TCIA manifest metadata to processed and Kaggle scans"
    ).parse_args()
    ensure_dirs()
    require_file(TCIA_MANIFEST, "TCIA manifest")
    series_csv = _series_csv()
    attach_processed(args.force, series_csv)
    attach_kaggle(args.force, series_csv)


if __name__ == "__main__":
    main()
