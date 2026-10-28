"""Local command-line workflow helper for the MCI-to-AD project.

This is not a cloud/LLM agent. It runs the project's Python scripts locally and
prints only file readiness and aggregate counts. ADNI data is never uploaded.

Examples (from the project folder):
  py -3.14 .\scripts\project_agent.py status
  py -3.14 .\scripts\project_agent.py demo
  py -3.14 .\scripts\project_agent.py cohort
  py -3.14 .\scripts\project_agent.py match
  py -3.14 .\scripts\project_agent.py manifest
  py -3.14 .\scripts\project_agent.py train
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
META = ROOT / "data" / "metadata"
SCRIPTS = ROOT / "scripts"


def csv_row_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return max(sum(1 for _ in csv.reader(handle)) - 1, 0)


def status() -> int:
    print("Local MCI-to-AD project status (no participant IDs or rows shown)\n")
    required = ["DXSUM.csv", "PTDEMOG.csv", "MMSE.csv", "ADAS.csv", "MRI_metadata.csv"]
    for filename in required:
        state = "present" if (RAW / filename).is_file() else "missing"
        print(f"{filename}: {state}")
    roster_state = "present" if (RAW / "ROSTER.csv").is_file() else "not found (optional unless RID mapping is needed)"
    print(f"ROSTER.csv: {roster_state}")

    cohort_path = META / "master_cohort_clinical_only.csv"
    candidate_path = META / "mri_scan_candidates_local.csv"
    manifest_path = META / "mri_training_manifest.csv"
    print(f"\nClinical cohort rows: {csv_row_count(cohort_path)}")
    print(f"MRI metadata candidate rows: {csv_row_count(candidate_path)}")
    phase_cohort = META / "cohort_adni2_3_36m.csv"
    phase_candidates = META / "mri_scan_candidates_adni2_3_local.csv"
    phase_manifest = META / "mri_training_manifest_adni2_3.csv"
    print(f"ADNI2/ADNI3 36-month cohort rows: {csv_row_count(phase_cohort)}")
    print(f"ADNI2/ADNI3 MRI candidate rows: {csv_row_count(phase_candidates)}")

    ready_nifti = 0
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                confirmed = (row.get("T1_CONFIRMED") or "").strip().casefold() in {"yes", "y", "true", "1"}
                image_path = (row.get("NIFTI_PATH") or "").strip()
                if not image_path:
                    continue
                resolved = Path(image_path)
                if not resolved.is_absolute():
                    resolved = ROOT / resolved
                if confirmed and resolved.is_file():
                    ready_nifti += 1
    print(f"Reviewed T1 NIfTI files in manifest: {ready_nifti}")
    phase_ready = 0
    if phase_manifest.is_file():
        with phase_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                confirmed = (row.get("T1_CONFIRMED") or "").strip().casefold() in {"yes", "y", "true", "1"}
                image_path = (row.get("NIFTI_PATH") or "").strip()
                if not image_path:
                    continue
                resolved = Path(image_path)
                if not resolved.is_absolute():
                    resolved = ROOT / resolved
                if confirmed and resolved.is_file():
                    phase_ready += 1
    print(f"Reviewed T1 NIfTI files in ADNI2/ADNI3 manifest: {phase_ready}")

    if ready_nifti < 20:
        print("\nNext: obtain and review more baseline T1 scans, then update the local manifest.")
        print("The training script currently stops below 20 scans; that is only a technical guard, not a robust sample-size target.")
        return 0
    print("\nThere are at least 20 reviewed NIfTI entries. Run the training script only after checking labels, dates, and one-scan-per-participant locally.")
    return 0


def run_script(script: str, script_args: list[str] | None = None) -> int:
    path = SCRIPTS / script
    if not path.is_file():
        print(f"Project script missing: {path}", file=sys.stderr)
        return 2
    result = subprocess.run([sys.executable, str(path), *(script_args or [])], cwd=ROOT, check=False)
    return int(result.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["status", "demo", "cohort", "cohort-adni23", "match", "manifest", "train",
                                              "match-adni23", "manifest-adni23", "train-adni23"])
    command = parser.parse_args().command
    if command == "status":
        return status()
    scripts = {
        "demo": "demo_synthetic.py",
        "cohort": "build_cohort_from_tables.py",
        "cohort-adni23": "build_adni2_3_cohort.py",
        "match": "match_mri_metadata.py",
        "manifest": "make_mri_manifest.py",
        "train": "train_multimodal.py",
    }
    if command == "match-adni23":
        return run_script("match_mri_metadata.py", [
            "--cohort", "data/metadata/cohort_adni2_3_36m.csv",
            "--output", "data/metadata/mri_scan_candidates_adni2_3_local.csv",
        ])
    if command == "manifest-adni23":
        return run_script("make_mri_manifest.py", [
            "--cohort", "data/metadata/cohort_adni2_3_36m.csv",
            "--candidates", "data/metadata/mri_scan_candidates_adni2_3_local.csv",
            "--output", "data/metadata/mri_training_manifest_adni2_3.csv",
        ])
    if command == "train-adni23":
        return run_script("train_multimodal.py", [
            "--manifest", "data/metadata/mri_training_manifest_adni2_3.csv",
            "--model-output", "models/multimodal_adni2_3_prototype.pt",
            "--report-output", "reports/multimodal_adni2_3_prototype_metrics.json",
        ])
    return run_script(scripts[command])


if __name__ == "__main__":
    raise SystemExit(main())
