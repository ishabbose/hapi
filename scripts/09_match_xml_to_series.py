#!/usr/bin/env python3
"""Step 09 — Match each nodule's CT SeriesInstanceUID to LIDC XML files."""

import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import common_parser
from hapi.config import (
    LIDC_XML_DIR,
    MASTER_MANIFEST_CSV,
    NODULE_XML_CSV,
    ensure_dirs,
    require_file,
)
from hapi.io_utils import should_skip
from hapi.xml_annotations import index_xml_by_series_uid


def main() -> None:
    args = common_parser("Match CT series to LIDC XML files").parse_args()
    ensure_dirs()
    if should_skip(NODULE_XML_CSV, args.force):
        return

    series_df = pd.read_csv(require_file(MASTER_MANIFEST_CSV, "master manifest (run 08)"))
    require_file(LIDC_XML_DIR, "XML dir (run 03)")
    xml_by_series = index_xml_by_series_uid(LIDC_XML_DIR)
    print(f"Indexed SeriesInstanceUIDs in XML: {len(xml_by_series)}")

    uid_col = "SeriesInstanceUID" if "SeriesInstanceUID" in series_df.columns else "series_instance_uid"
    rows = []
    for _, row in series_df.iterrows():
        series_uid = str(row[uid_col])
        candidates = xml_by_series.get(series_uid, [])
        if not candidates:
            status = "NO_XML_MATCH"
        elif len(candidates) == 1:
            status = "ONE_XML_MATCH"
        else:
            status = "MULTIPLE_XML_MATCHES"
        rows.append(
            {
                "subset_nodule_id": row["subset_nodule_id"],
                "patient_id": row["patient_id"],
                "native_nodule_id": row["native_nodule_id"],
                "SeriesInstanceUID": series_uid,
                "xml_count": len(candidates),
                "xml_paths": " || ".join(candidates),
                "xml_match_status": status,
            }
        )

    result = pd.DataFrame(rows)
    print(result["xml_match_status"].value_counts().to_string())
    result.to_csv(NODULE_XML_CSV, index=False)
    print(f"Saved {NODULE_XML_CSV}")


if __name__ == "__main__":
    main()
