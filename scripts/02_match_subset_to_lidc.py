#!/usr/bin/env python3
"""Step 02 — SHA-256 match subset nodules to native LIDC patient/nodule folders."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import common_parser
from hapi.config import NODULE_TO_LIDC_CSV, ORIGINAL_ZIP, SUBSET_ZIP, ensure_dirs, require_file
from hapi.io_utils import should_skip
from hapi.matching import match_subset_to_lidc


def main() -> None:
    args = common_parser("Match 2,000-image subset to original LIDC PNGs").parse_args()
    ensure_dirs()
    if should_skip(NODULE_TO_LIDC_CSV, args.force):
        return
    require_file(SUBSET_ZIP, "subset ZIP")
    require_file(ORIGINAL_ZIP, "original LIDC ZIP")
    mapping = match_subset_to_lidc(SUBSET_ZIP, ORIGINAL_ZIP)
    mapping.to_csv(NODULE_TO_LIDC_CSV, index=False)
    print(f"Saved {NODULE_TO_LIDC_CSV} ({len(mapping)} nodules)")


if __name__ == "__main__":
    main()
