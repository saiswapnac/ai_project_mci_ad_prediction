r"""Local command-line workflow helper for the MCI-to-AD project.

This is not a cloud/LLM agent. It runs the project's Python scripts locally and
prints only file readiness, aggregate counts, and evaluation summaries.
ADNI data is never uploaded.

Examples (from the project folder):
  py -3.14 .\scripts\project_agent.py status
  py -3.14 .\scripts\project_agent.py cohort
  py -3.14 .\scripts\project_agent.py convert
  py -3.14 .\scripts\project_agent.py train-all
  py -3.14 .\scripts\project_agent.py figures
  py -3.14 .\scripts\project_agent.py evaluate
  py -3.14 .\scripts\project_agent.py demo
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
META = ROOT / "data" / "metadata"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
SCRIPTS = ROOT / "scripts"


def csv_row_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return max(sum(1 for _ in csv.reader(handle)) - 1, 0)


def status() -> int:
    print("=== MCI-to-Alzheimer's Progression Project Status ===\n")
    required = ["DXSUM.csv", "PTDEMOG.csv", "MMSE.csv", "ADAS.csv"]
    for filename in required:
        state = "present" if (RAW / filename).is_file() else "missing"
        print(f"  {filename:16s}: {state}")
    roster_state = "present" if (RAW / "ROSTER.csv").is_file() else "optional"
    print(f"  {'ROSTER.csv':16s}: {roster_state}")

    cohort_path = META / "master_cohort_clinical_only.csv"
    manifest_path = META / "mri_training_manifest.csv"
    cache_path = ROOT / "data" / "processed" / "mri_preprocessed_cache.pt"
    
    print(f"\nCohort & Data Readiness:")
    print(f"  Clinical cohort rows      : {csv_row_count(cohort_path)}")
    print(f"  Reviewed T1 NIfTI scans   : {csv_row_count(manifest_path)}")
    print(f"  MRI preprocessed tensor cache: {'present' if cache_path.is_file() else 'not built'}")

    # Check trained models
    print(f"\nTrained Model Weights:")
    models_to_check = [
        ("Clinical-only MLP", "clinical-only_mlp.pt"),
        ("MRI-only 3D CNN", "mri-only_3d_cnn.pt"),
        ("Combined Multimodal", "combined_cnn_and_mlp.pt"),
    ]
    for label, fname in models_to_check:
        state = "trained & saved" if (MODELS / fname).is_file() else "not trained"
        print(f"  {label:24s}: {state}")

    # Check Figures
    print(f"\nGenerated Report Figures:")
    figs = list(FIGURES.glob("*.png"))
    print(f"  Total publication figures : {len(figs)} in reports/figures/")
    for f in sorted(figs):
        print(f"    - {f.name}")

    # Check Evaluation Results
    eval_json = REPORTS / "evaluation_metrics.json"
    comp_csv = REPORTS / "model_comparison_table.csv"
    if comp_csv.is_file():
        print(f"\nLatest Comparative Evaluation Metrics:")
        with comp_csv.open("r", encoding="utf-8") as f:
            for line in f:
                print("  " + line.strip())

    print("\nAvailable workflow commands:")
    print("  py -3.14 .\\scripts\\project_agent.py cohort      (rebuild cohort from ADNI tables)")
    print("  py -3.14 .\\scripts\\project_agent.py convert     (convert DICOM scans to NIfTI)")
    print("  py -3.14 .\\scripts\\project_agent.py train-all   (train Clinical MLP, MRI CNN, Combined)")
    print("  py -3.14 .\\scripts\\project_agent.py figures     (generate Figures 1-7 in reports/figures/)")
    print("  py -3.14 .\\scripts\\project_agent.py evaluate    (display sex-stratified evaluation)")
    return 0


def evaluate_summary() -> int:
    metrics_path = REPORTS / "evaluation_metrics.json"
    if not metrics_path.is_file():
        print("No evaluation metrics found. Run train-all first.")
        return 1
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("=========================================================================================")
    print("           MCI-TO-ALZHEIMER'S PROGRESSION: HELD-OUT TEST EVALUATION METRICS              ")
    print("=========================================================================================\n")

    for model_name, res in data.items():
        om = res["overall"]
        cm = om.get("confusion_matrix", {})
        print(f"--- Model: {model_name} (Overall n={om['n']}, pMCI={om['n_pMCI']}, sMCI={om['n_sMCI']}) ---")
        print(f"  Accuracy           : {om['accuracy']:.4f}")
        print(f"  Sensitivity (Recall): {om['sensitivity']:.4f}")
        print(f"  Specificity        : {om['specificity']:.4f}")
        print(f"  F1-Score           : {om['f1']:.4f}")
        print(f"  ROC-AUC            : {om['roc_auc'] if om['roc_auc'] is not None else 'N/A'}")
        print(f"  False-Negative Rate: {om['false_negative_rate']:.4f}")
        print(f"  Confusion Matrix   : TN={cm.get('TN')}, FP={cm.get('FP')}, FN={cm.get('FN')}, TP={cm.get('TP')}")

        by_sex = res.get("by_sex", {})
        if by_sex:
            print("  Sex-Stratified Performance:")
            for sex, sm in by_sex.items():
                print(f"    * {sex:6s} (n={sm['n']:2d}, pMCI={sm['n_pMCI']}, sMCI={sm['n_sMCI']}): "
                      f"Acc={sm['accuracy']:.3f} | Sens={sm['sensitivity']:.3f} | "
                      f"Spec={sm['specificity']:.3f} | F1={sm['f1']:.3f} | "
                      f"AUC={sm['roc_auc'] if sm['roc_auc'] is not None else 'N/A'} | "
                      f"FNR={sm['false_negative_rate']:.3f}")
        print()
    return 0


def run_script(script: str, script_args: list[str] | None = None) -> int:
    path = SCRIPTS / script
    if not path.is_file():
        print(f"Project script missing: {path}", file=sys.stderr)
        return 2
    result = subprocess.run([sys.executable, "-u", str(path), *(script_args or [])], cwd=ROOT, check=False)
    return int(result.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=[
        "status", "demo", "cohort", "cohort-adni23", "match", "manifest",
        "convert", "train", "train-all", "figures", "evaluate",
        "match-adni23", "manifest-adni23", "train-adni23"
    ])
    command = parser.parse_args().command
    if command == "status":
        return status()
    if command == "evaluate":
        return evaluate_summary()

    scripts = {
        "demo": "demo_synthetic.py",
        "cohort": "build_cohort_from_tables.py",
        "cohort-adni23": "build_adni2_3_cohort.py",
        "match": "match_mri_metadata.py",
        "manifest": "make_mri_manifest.py",
        "convert": "convert_all_manifest_dicoms.py",
        "train": "train_multimodal.py",
        "train-all": "train_and_evaluate_all_models.py",
        "figures": "generate_all_figures.py",
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
