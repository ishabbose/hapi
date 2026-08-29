#!/usr/bin/env python3
"""Step 05 — Read SeriesInstanceUIDs from the TCIA .tcia manifest."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import common_parser
from hapi.config import ARTIFACTS_DIR, TCIA_MANIFEST, ensure_dirs, require_file
from hapi.io_utils import should_skip
from hapi.tcia import parse_series_uids


def main() -> None:
    args = common_parser("Parse TCIA LIDC-IDRI manifest UIDs").parse_args()
    ensure_dirs()
    out = ARTIFACTS_DIR / "tcia_series_uids.txt"
    if should_skip(out, args.force):
        return
    require_file(TCIA_MANIFEST, "TCIA manifest")
    uids = parse_series_uids(TCIA_MANIFEST)
    out.write_text("\n".join(uids) + "\n")
    print(f"Unique SeriesInstanceUIDs: {len(uids)}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
