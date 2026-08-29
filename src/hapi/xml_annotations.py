"""LIDC XML annotation inventory and SeriesInstanceUID lookup."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from hapi.io_utils import local_tag


def list_xml_files(xml_dir: Path) -> list[Path]:
    xml_files: list[Path] = []
    for root, dirs, files in os.walk(xml_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__MACOSX"]
        for file in files:
            if file.lower().endswith(".xml"):
                xml_files.append(Path(root) / file)
    return sorted(xml_files)


def build_xml_inventory(xml_dir: Path) -> pd.DataFrame:
    records = []
    xml_files = list_xml_files(xml_dir)
    print(f"XML files found: {len(xml_files)}")

    for i, xml_path in enumerate(xml_files, start=1):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            series_uids: list[str] = []
            study_uids: list[str] = []
            patient_ids: list[str] = []
            malignancies: list[int] = []
            nodule_count = 0

            for elem in root.iter():
                tag = local_tag(elem.tag)
                value = (elem.text or "").strip()
                if tag == "SeriesInstanceUid" and value:
                    series_uids.append(value)
                elif tag.lower() == "studyinstanceuid" and value:
                    study_uids.append(value)
                elif tag.lower() == "patientid" and value:
                    patient_ids.append(value)
                elif tag.lower() == "malignancy" and value:
                    try:
                        malignancies.append(int(value))
                    except ValueError:
                        pass
                elif tag == "unblindedReadNodule":
                    nodule_count += 1

            records.append(
                {
                    "xml_path": str(xml_path),
                    "xml_filename": xml_path.name,
                    "patient_id_in_xml": patient_ids[0] if patient_ids else "",
                    "series_instance_uid": series_uids[0] if series_uids else "",
                    "study_instance_uid": study_uids[0] if study_uids else "",
                    "nodule_count": nodule_count,
                    "malignancy_ratings": str(malignancies),
                    "num_malignancy_values": len(malignancies),
                }
            )
        except Exception as exc:
            records.append(
                {
                    "xml_path": str(xml_path),
                    "xml_filename": xml_path.name,
                    "patient_id_in_xml": "",
                    "series_instance_uid": "",
                    "study_instance_uid": "",
                    "nodule_count": -1,
                    "malignancy_ratings": "",
                    "num_malignancy_values": -1,
                    "parse_error": str(exc),
                }
            )
        if i % 200 == 0:
            print(f"Processed {i} / {len(xml_files)} XML files")

    return pd.DataFrame(records)


def index_xml_by_series_uid(xml_dir: Path) -> dict[str, list[str]]:
    xml_by_series: dict[str, list[str]] = {}
    for xml_path in list_xml_files(xml_dir):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for elem in root.iter():
                tag = local_tag(elem.tag).lower()
                if tag == "seriesinstanceuid":
                    value = (elem.text or "").strip()
                    if value:
                        xml_by_series.setdefault(value, []).append(str(xml_path))
        except ET.ParseError:
            continue
    return xml_by_series
