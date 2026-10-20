"""Convert one reviewed DICOM series folder to compressed NIfTI using dcm2niix.

Example:
  py -3.14 .\scripts\convert_dicom_series.py "D:\\ADNI\\one_series" \
      --output .\data\processed\nifti

Organize/pass one scan series at a time. Do not pass the entire ADNI directory.
Install dcm2niix separately and ensure it is available on PATH.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("series_dir", type=Path, help="Folder containing a single reviewed DICOM series")
    parser.add_argument("--output", type=Path, default=Path("data/processed/nifti"))
    args = parser.parse_args()
    converter = shutil.which("dcm2niix")
    if not converter:
        print("dcm2niix was not found on PATH. Install it, reopen PowerShell, and retry.", file=sys.stderr)
        return 2
    if not args.series_dir.is_dir():
        print("The input must be an existing folder for one scan series.", file=sys.stderr)
        return 2
    args.output.mkdir(parents=True, exist_ok=True)
    command = [converter, "-z", "y", "-b", "y", "-ba", "y", "-f", "%p_%t", "-o",
               str(args.output.resolve()), str(args.series_dir.resolve())]
    result = subprocess.run(command, check=False)
    if result.returncode:
        print("dcm2niix conversion failed; review its local output above.", file=sys.stderr)
        return result.returncode
    print("Conversion complete. Confirm the output is the intended T1 series before modeling.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
