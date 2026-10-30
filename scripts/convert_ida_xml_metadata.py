"""Flatten downloaded ADNI IDA XML metadata archives into a local CSV.

Run from the project folder:
  py -3.14 .\\scripts\\convert_ida_xml_metadata.py

The output contains scan-level metadata and participant crosswalk fields; keep
it on approved local storage. It does not extract or modify the DICOM archives.
"""
from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "metadata"
DEFAULT_OUTPUT = ROOT / "data" / "raw" / "MRI_metadata_from_xml.csv"


def local_path(value: Path) -> Path:
    return value if value.is_absolute() else ROOT / value


def text_for(root: ET.Element, field_name: str) -> str:
    """Read text for a named XML element without depending on its namespace."""
    values = []
    wanted = field_name.casefold()
    for element in root.iter():
        if element.tag.split("}")[-1].casefold() == wanted:
            value = " ".join("".join(element.itertext()).split())
            if value and value not in values:
                values.append(value)
    return " | ".join(values)


def parse_xml(xml_bytes: bytes, source_name: str) -> dict[str, str] | None:
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return None

    image_id = text_for(root, "imageUID")
    # Some IDA metadata exports place the image identifier in the XML filename.
    if not image_id:
        match = re.search(r"S\d+I\d+", Path(source_name).stem, re.IGNORECASE)
        image_id = match.group(0) if match else ""
    subject = text_for(root, "subjectIdentifier")
    if not image_id or not subject:
        return None

    description = text_for(root, "description")
    return {
        "Image Data ID": image_id,
        "Subject": subject,
        "Group": text_for(root, "researchGroup"),
        "Sex": text_for(root, "subjectSex"),
        "Age": text_for(root, "subjectAge"),
        "Visit": text_for(root, "visitIdentifier"),
        "Modality": text_for(root, "modality"),
        "Description": description,
        "Type": "DICOM",
        "Acq Date": text_for(root, "dateAcquired"),
        "Format": "DCM",
        "Downloaded": "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT,
                        help="Folder containing dataset_IDA_Metadata*.zip archives")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output metadata CSV; defaults to a new file so the existing CSV is preserved")
    args = parser.parse_args()
    input_dir = local_path(args.input_dir)
    output_path = local_path(args.output)
    archives = sorted(input_dir.glob("dataset_IDA_Metadata*.zip"))
    if not archives:
        raise FileNotFoundError(f"No dataset_IDA_Metadata*.zip archives found in {input_dir}")

    records: dict[str, dict[str, str]] = {}
    xml_files = 0
    ignored_xml = 0
    duplicate_image_records = 0
    for archive in archives:
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if info.is_dir() or not info.filename.casefold().endswith(".xml"):
                    continue
                xml_files += 1
                record = parse_xml(zf.read(info), info.filename)
                if record is None:
                    ignored_xml += 1
                    continue
                key = record["Image Data ID"].strip().casefold()
                previous = records.get(key)
                if previous is None:
                    records[key] = record
                else:
                    duplicate_image_records += 1
                    # Retain the duplicate with the most populated metadata fields.
                    if sum(bool(v) for v in record.values()) > sum(bool(v) for v in previous.values()):
                        records[key] = record

    if not records:
        raise ValueError("No usable image metadata records were found in the XML archives.")
    result = pd.DataFrame(records.values())
    result["Acq Date"] = pd.to_datetime(result["Acq Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    result = result.sort_values(["Subject", "Acq Date", "Image Data ID"], kind="stable")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print("IDA XML metadata conversion complete (aggregate counts only):")
    print("XML files scanned:", xml_files)
    print("Unique image metadata records:", len(result))
    print("Duplicate image records merged:", duplicate_image_records)
    print("XML files without usable scan fields:", ignored_xml)
    print("Acquisition dates parsed:", int(result["Acq Date"].notna().sum()))
    print("Saved local metadata CSV:", output_path)
    print("Participant identifiers remain only in the local CSV; do not share it publicly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
