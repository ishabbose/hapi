#!/usr/bin/env python3
"""Step 03 — Download official LIDC-IDRI XML annotations from TCIA wiki."""

import urllib.request
import zipfile

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import common_parser
from hapi.config import LIDC_XML_DIR, LIDC_XML_ZIP, LIDC_XML_ZIP_URL, ensure_dirs
from hapi.io_utils import should_skip


def main() -> None:
    args = common_parser("Download and extract LIDC XML annotations").parse_args()
    ensure_dirs()
    marker = LIDC_XML_DIR / ".extracted"
    if should_skip(marker, args.force):
        return

    if not LIDC_XML_ZIP.exists():
        print(f"Downloading {LIDC_XML_ZIP_URL}")
        urllib.request.urlretrieve(LIDC_XML_ZIP_URL, LIDC_XML_ZIP)
    else:
        print(f"Using existing ZIP: {LIDC_XML_ZIP}")

    LIDC_XML_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(LIDC_XML_ZIP, "r") as z:
        z.extractall(LIDC_XML_DIR)
    marker.write_text("ok\n")
    print(f"Extracted XML to {LIDC_XML_DIR}")


if __name__ == "__main__":
    main()
