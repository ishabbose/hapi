"""TCIA LIDC-IDRI manifest parsing and NBIA series metadata."""

from __future__ import annotations

import json
import math
import tempfile
import time
import zipfile
from pathlib import Path

import pandas as pd

from hapi.config import (
    MASTER_MANIFEST_CSV,
    NODULE_SINGLE_CT_CSV,
    NODULE_TO_LIDC_CSV,
    TCIA_GET_SERIES_URL,
)


def parse_tcia_manifest(manifest_path: Path) -> tuple[dict[str, str], list[str]]:
    """Return (header key=value fields, unique SeriesInstanceUIDs).

    A `.tcia` file is a download basket: header metadata plus a UID list.
    It does not contain per-series scanner/patient fields.
    """
    text = manifest_path.read_text(encoding="utf-8", errors="ignore")
    header: dict[str, str] = {}
    series_uids: list[str] = []
    started = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("ListOfSeriesToDownload="):
            started = True
            remainder = line.split("=", 1)[1].strip()
            if remainder:
                series_uids.append(remainder)
            continue
        if not started:
            if "=" in line:
                key, value = line.split("=", 1)
                header[key.strip()] = value.strip()
            continue
        if line.startswith("1."):
            series_uids.append(line)
    return header, sorted(set(series_uids))


def parse_series_uids(manifest_path: Path) -> list[str]:
    _, uids = parse_tcia_manifest(manifest_path)
    return uids


def _json_safe(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (ValueError, AttributeError):
            return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def load_series_by_uid(series_csv: Path | None) -> dict[str, dict]:
    series_by_uid: dict[str, dict] = {}
    if series_csv is None or not series_csv.exists():
        return series_by_uid
    series_df = pd.read_csv(series_csv)
    if "SeriesInstanceUID" not in series_df.columns:
        raise ValueError(f"{series_csv} is missing SeriesInstanceUID")
    series_df = series_df.drop_duplicates(subset=["SeriesInstanceUID"])
    series_df["SeriesInstanceUID"] = series_df["SeriesInstanceUID"].astype(str)
    for uid, row in series_df.set_index("SeriesInstanceUID", drop=False).iterrows():
        series_by_uid[str(uid)] = {
            col: _json_safe(row[col]) for col in series_df.columns
        }
    return series_by_uid


def tcia_fields_for_uid(
    series_uid: str,
    header: dict[str, str],
    uid_set: set[str],
    series_by_uid: dict[str, dict],
    manifest_path: Path,
) -> tuple[dict, dict | None]:
    series_uid = (series_uid or "").strip()
    manifest_block = {
        "manifest_file": str(manifest_path),
        **header,
        "in_list_of_series_to_download": series_uid in uid_set if series_uid else False,
    }
    return manifest_block, series_by_uid.get(series_uid) if series_uid else None


def _attachment_row(
    *,
    subset_nodule_id: str,
    patient_id: str,
    native_nodule_id: str,
    series_uid: str,
    metadata_path: str,
    manifest_block: dict,
    series_fields: dict | None,
    header: dict[str, str],
    source: str,
) -> dict:
    return {
        "source": source,
        "subset_nodule_id": subset_nodule_id,
        "patient_id": patient_id,
        "native_nodule_id": native_nodule_id,
        "SeriesInstanceUID": series_uid,
        "metadata_path": metadata_path,
        "in_tcia_manifest": bool(manifest_block.get("in_list_of_series_to_download")),
        "has_tcia_series_metadata": series_fields is not None,
        **{f"tcia_{k}": v for k, v in header.items()},
    }


def attach_tcia_metadata_to_processed(
    processed_dir: Path,
    manifest_path: Path,
    series_csv: Path | None = None,
) -> pd.DataFrame:
    """Write TCIA basket + matching series fields into each nodule metadata.json."""
    header, uids = parse_tcia_manifest(manifest_path)
    uid_set = set(uids)
    series_by_uid = load_series_by_uid(series_csv)

    case_dirs = sorted(
        p for p in processed_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    if not case_dirs:
        raise FileNotFoundError(f"No nodule folders under {processed_dir}")

    rows = []
    for case_dir in case_dirs:
        meta_path = case_dir / "metadata.json"
        if not meta_path.exists():
            print(f"WARNING: missing metadata.json in {case_dir}")
            continue
        metadata = json.loads(meta_path.read_text())
        series_uid = str(metadata.get("SeriesInstanceUID", "")).strip()
        manifest_block, series_fields = tcia_fields_for_uid(
            series_uid, header, uid_set, series_by_uid, manifest_path
        )
        metadata["tcia_manifest"] = manifest_block
        metadata["tcia_series"] = series_fields
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
        rows.append(
            {
                k: v
                for k, v in _attachment_row(
                    subset_nodule_id=str(metadata.get("subset_nodule_id", case_dir.name)),
                    patient_id=str(metadata.get("patient_id", "")),
                    native_nodule_id=str(metadata.get("native_nodule_id", "")),
                    series_uid=series_uid,
                    metadata_path=str(meta_path),
                    manifest_block=manifest_block,
                    series_fields=series_fields,
                    header=header,
                    source="processed_325_nodules",
                ).items()
                if k != "source"
            }
        )

    return pd.DataFrame(rows)


def load_subset_nodule_identity() -> dict[str, dict]:
    """subset_nodule_id -> patient / native nodule / SeriesInstanceUID."""
    records: dict[str, dict] = {}
    for path in (NODULE_TO_LIDC_CSV, NODULE_SINGLE_CT_CSV, MASTER_MANIFEST_CSV):
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "subset_nodule_id" not in df.columns:
            continue
        for _, row in df.iterrows():
            nid = str(row["subset_nodule_id"])
            uid = ""
            if "SeriesInstanceUID" in row.index and pd.notna(row["SeriesInstanceUID"]):
                uid = str(row["SeriesInstanceUID"]).strip()
            incoming = {
                "subset_nodule_id": nid,
                "patient_id": "" if pd.isna(row.get("patient_id")) else str(row.get("patient_id", "")),
                "native_nodule_id": (
                    ""
                    if pd.isna(row.get("native_nodule_id"))
                    else str(row.get("native_nodule_id", ""))
                ),
                "SeriesInstanceUID": uid,
            }
            current = records.get(nid)
            if current is None:
                records[nid] = incoming
            elif uid and not current.get("SeriesInstanceUID"):
                records[nid] = incoming
            elif uid:
                records[nid]["SeriesInstanceUID"] = uid
                if incoming["patient_id"]:
                    records[nid]["patient_id"] = incoming["patient_id"]
                if incoming["native_nodule_id"]:
                    records[nid]["native_nodule_id"] = incoming["native_nodule_id"]
    return records


def upsert_files_in_zip(zip_path: Path, members: dict[str, bytes]) -> None:
    """Replace or add named members without changing other zip entries."""
    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".zip", dir=zip_path.parent)
    tmp_path = Path(tmp_name)
    import os

    os.close(tmp_fd)
    try:
        with zipfile.ZipFile(zip_path, "r") as zin, zipfile.ZipFile(
            tmp_path, "w"
        ) as zout:
            skip = set(members)
            for item in zin.infolist():
                if item.filename in skip:
                    continue
                zout.writestr(item, zin.read(item.filename))
            for name, data in members.items():
                zout.writestr(name, data, compress_type=zipfile.ZIP_DEFLATED)
        tmp_path.replace(zip_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def attach_tcia_metadata_to_kaggle(
    extract_dir: Path,
    zip_path: Path,
    manifest_path: Path,
    series_csv: Path | None = None,
    identity: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """Write TCIA fields onto extracted Kaggle nodule folders and into the ZIP."""
    from hapi.volumes import discover_volumes, volume_id_for_path

    header, uids = parse_tcia_manifest(manifest_path)
    uid_set = set(uids)
    series_by_uid = load_series_by_uid(series_csv)
    identity = identity if identity is not None else load_subset_nodule_identity()

    volumes = discover_volumes(extract_dir)
    if not volumes:
        raise FileNotFoundError(f"No PNG nodule folders under {extract_dir}")

    zip_prefix_by_id: dict[str, str] = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.endswith("/") or not name.lower().endswith(".png"):
                continue
            vol_id = volume_id_for_path(Path(name))
            if vol_id not in zip_prefix_by_id:
                parent = name.replace("\\", "/").rsplit("/", 1)[0]
                zip_prefix_by_id[vol_id] = parent + "/"

    zip_members: dict[str, bytes] = {}
    rows = []
    for vol_id, image_files in sorted(volumes.items()):
        case_dir = Path(image_files[0]).parent
        meta_path = case_dir / "metadata.json"
        info = identity.get(vol_id, {})
        series_uid = str(info.get("SeriesInstanceUID", "")).strip()
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text())
            if not series_uid:
                series_uid = str(metadata.get("SeriesInstanceUID", "")).strip()
        else:
            metadata = {}
        metadata.update(
            {
                "subset_nodule_id": info.get("subset_nodule_id", vol_id),
                "patient_id": info.get("patient_id", metadata.get("patient_id", "")),
                "native_nodule_id": info.get(
                    "native_nodule_id", metadata.get("native_nodule_id", "")
                ),
                "SeriesInstanceUID": series_uid,
                "image_slice_count": len(image_files),
                "source_zip": str(zip_path),
            }
        )
        manifest_block, series_fields = tcia_fields_for_uid(
            series_uid, header, uid_set, series_by_uid, manifest_path
        )
        metadata["tcia_manifest"] = manifest_block
        metadata["tcia_series"] = series_fields
        encoded = (json.dumps(metadata, indent=2) + "\n").encode("utf-8")
        meta_path.write_bytes(encoded)
        prefix = zip_prefix_by_id.get(vol_id)
        if prefix:
            zip_members[prefix + "metadata.json"] = encoded
        rows.append(
            _attachment_row(
                subset_nodule_id=str(metadata["subset_nodule_id"]),
                patient_id=str(metadata.get("patient_id", "")),
                native_nodule_id=str(metadata.get("native_nodule_id", "")),
                series_uid=series_uid,
                metadata_path=str(meta_path),
                manifest_block=manifest_block,
                series_fields=series_fields,
                header=header,
                source="kaggle_dataset_2000",
            )
        )

    if zip_members:
        print(f"Writing {len(zip_members)} metadata.json files into {zip_path}")
        upsert_files_in_zip(zip_path, zip_members)
    return pd.DataFrame(rows)


def fetch_series_metadata(
    series_uids: list[str],
    sleep_s: float = 0.15,
):
    from io import StringIO

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
