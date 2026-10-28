"""Inspect and convert selected ADNI .rda tables to CSV using pyreadr.

Usage:
  py convert_adni_rda.py "C:\\path\\to\\extracted\\data"
  py convert_adni_rda.py "C:\\path\\to\\extracted\\data" --output "C:\\path\\to\\mci_ad_prediction\\data\\raw"
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

TARGETS = {"dxsum", "ptdemog", "mmse", "adas"}
TARGETS.add("roster")

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Folder containing the extracted .rda files")
    parser.add_argument("--output", type=Path, default=None, help="CSV output folder (default: <input_dir>/converted_csv)")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing CSV outputs; by default existing files are skipped")
    args = parser.parse_args()
    if not args.input_dir.is_dir():
        parser.error(f"Input folder does not exist: {args.input_dir}")
    try:
        import pyreadr
    except ImportError:
        print("pyreadr is not installed. Install it with: py -m pip install pyreadr", file=sys.stderr)
        return 2

    output_dir = args.output or args.input_dir / "converted_csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    files = [p for p in args.input_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".rda", ".rdata"} and not p.stem.lower().startswith("rmt_") and any(target in p.stem.lower() for target in TARGETS)]
    if not files:
        print("No matching DXSUM/PTDEMOG/MMSE/ADAS .rda files found. Check the input folder and exact filenames.")
        return 1

    for path in sorted(files):
        print(f"\nFILE: {path.name}")
        try:
            objects = pyreadr.read_r(str(path))
        except Exception as exc:
            print(f"  Could not read file: {exc}")
            continue
        if not objects:
            print("  No data-frame objects found.")
            continue
        for obj_name, frame in objects.items():
            if not hasattr(frame, "to_csv"):
                print(f"  Object {obj_name!r} is not a table; skipped.")
                continue
            suffix = "" if len(objects) == 1 else f"_{obj_name}"
            dest = output_dir / f"{path.stem}{suffix}.csv"
            if dest.exists() and not args.overwrite:
                print(f"  CSV already exists: {dest}; skipped (use --overwrite to replace)")
                continue
            frame.to_csv(dest, index=False)
            print(f"  Object: {obj_name!r}; shape: {frame.shape}")
            print("  Columns:", ", ".join(map(str, frame.columns)))
            print(f"  CSV: {dest}")
    print(f"\nFinished. Review the printed object names/columns and CSVs in: {output_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())


