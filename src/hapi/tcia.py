"""TCIA LIDC-IDRI manifest parsing and NBIA series metadata."""

from __future__ import annotations

import time
from pathlib import Path

from hapi.config import TCIA_GET_SERIES_URL


def parse_series_uids(manifest_path: Path) -> list[str]:
    text = manifest_path.read_text(encoding="utf-8", errors="ignore")
    series_uids: list[str] = []
    started = False
    for line in text.splitlines():
        line = line.strip()
        if line == "ListOfSeriesToDownload=":
            started = True
            continue
        if started and line.startswith("1.3.6.1.4.1.14519."):
            series_uids.append(line)
    return sorted(set(series_uids))


def fetch_series_metadata(
    series_uids: list[str],
    sleep_s: float = 0.15,
):
    from io import StringIO

    import pandas as pd
    import requests

    rows = []
    failed: list[tuple[str, str]] = []

    for i, uid in enumerate(series_uids, start=1):
        try:
            response = requests.get(
                TCIA_GET_SERIES_URL,
                params={"SeriesInstanceUID": uid, "format": "csv"},
                timeout=120,
            )
            response.raise_for_status()
            text = response.text.strip()
            if not text:
                failed.append((uid, "EMPTY_RESPONSE"))
                continue
            df_one = pd.read_csv(StringIO(text))
            if df_one.empty:
                failed.append((uid, "EMPTY_TABLE"))
                continue
            rows.append(df_one)
        except Exception as exc:
            failed.append((uid, f"{type(exc).__name__}: {exc}"))

        if i % 50 == 0 or i == len(series_uids):
            print(f"Queried {i} / {len(series_uids)}")
        time.sleep(sleep_s)

    if not rows:
        raise RuntimeError("No TCIA metadata was returned.")

    tcia = pd.concat(rows, ignore_index=True)
    if "SeriesInstanceUID" in tcia.columns:
        tcia = tcia.drop_duplicates(subset=["SeriesInstanceUID"])
    print(f"Failed UIDs: {len(failed)}")
    return tcia
