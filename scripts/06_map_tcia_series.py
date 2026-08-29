#!/usr/bin/env python3
"""Step 06 — Query TCIA NBIA getSeries for PatientID / modality metadata.

This step hits the public TCIA API (~1,300 series). Safe to re-run; use --force
to overwrite the cached CSV.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import common_parser
from hapi.config import ARTIFACTS_DIR, TCIA_MANIFEST, TCIA_SERIES_CSV, ensure_dirs, require_file
from hapi.io_utils import should_skip
from hapi.tcia import fetch_series_metadata, parse_series_uids


def main() -> None:
    args = common_parser("Map TCIA SeriesInstanceUIDs to PatientIDs").parse_args()
    ensure_dirs()
    if should_skip(TCIA_SERIES_CSV, args.force):
        return
    require_file(TCIA_MANIFEST, "TCIA manifest")
    uids_file = ARTIFACTS_DIR / "tcia_series_uids.txt"
    if uids_file.exists():
        uids = [line.strip() for line in uids_file.read_text().splitlines() if line.strip()]
    else:
        uids = parse_series_uids(TCIA_MANIFEST)
    print(f"Querying {len(uids)} series...")
    tcia = fetch_series_metadata(uids)
    tcia.to_csv(TCIA_SERIES_CSV, index=False)
    print(f"Saved {TCIA_SERIES_CSV} ({len(tcia)} rows)")


if __name__ == "__main__":
    main()
