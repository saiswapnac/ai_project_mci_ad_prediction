"""Live Demonstration Script: Multimodal Prediction on Held-Out Test Participants.

Loads the trained Multimodal Fusion Network and demonstrates live end-to-end inference
on test set participants using their baseline clinical data and 3D T1-weighted MRI volume.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "metadata" / "mri_training_manifest.csv"
CACHE_PATH = ROOT / "data" / "processed" / "mri_preprocessed_cache.pt"
MODEL_PATH = ROOT / "models" / "combined_cnn_and_mlp.pt"
CLINICAL_COLS = ["age_years_approx", "education_years", "mmse_score", "adas_score", "sex_binary"]
SEED = 42

from train_and_evaluate_all_models import MultimodalFusionNet


def main():
    print("=" * 80)
    print("  LIVE DEMONSTRATION: MULTIMODAL 36-MONTH ALZHEIMER'S PROGRESSION PREDICTION")
    print("=" * 80)

    if not MANIFEST.is_file() or not CACHE_PATH.is_file() or not MODEL_PATH.is_file():
        print("Required manifest, MRI cache, or trained model missing. Run train-all first.")
        return

    # Load manifest and cache
    df = pd.read_csv(MANIFEST)
    df.columns = df.columns.str.strip().str.lower()
    df = df[df["t1_confirmed"].astype(str).str.strip().str.casefold().isin({"yes", "y", "true", "1"})]
    df = df.drop_duplicates(subset=["ptid"], keep="first").reset_index(drop=True)

    # Prepare features
    clin = df[["age_years_approx", "education_years", "mmse_score", "adas_score"]].apply(pd.to_numeric, errors="coerce")
    clin["sex_binary"] = df["sex"].str.strip().str.casefold().map({"female": 0.0, "male": 1.0}).astype(float)
    y = df["label"].astype(int).to_numpy()

    # Reconstruct exact held-out split
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=SEED, stratify=y)

    train_clin = clin.iloc[train_idx]
    medians = train_clin.median().fillna(0.0)
    means = train_clin.fillna(medians).mean()
    stds = train_clin.fillna(medians).std().replace(0, 1).fillna(1)
    x_clin_norm = ((clin.fillna(medians) - means) / stds).to_numpy(dtype=np.float32)

    x_img = torch.load(CACHE_PATH, weights_only=True)
    x_cli = torch.from_numpy(x_clin_norm)

    # Load Model
    model = MultimodalFusionNet(clinical_size=5)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()

    print(f"\n[OK] Loaded trained Multimodal Fusion Network weights: {MODEL_PATH.name}")
    print(f"[OK] Total participants: {len(df)} | Held-out test set: {len(test_idx)} participants")

    # Select representative test cases
    test_df = df.iloc[test_idx].reset_index(drop=True)
    test_img = x_img[test_idx]
    test_cli = x_cli[test_idx]
    test_y = y[test_idx]

    # Find one sMCI and one pMCI
    cases = []
    # Pick first sMCI
    for i in range(len(test_y)):
        if test_y[i] == 0:
            cases.append(i)
            break
    # Pick first pMCI
    for i in range(len(test_y)):
        if test_y[i] == 1:
            cases.append(i)
            break

    print("\n" + "-" * 80)
    print("DEMO: EVALUATING HELD-OUT PARTICIPANTS THROUGH THE MULTIMODAL PIPELINE")
    print("-" * 80)

    for case_num, idx in enumerate(cases, 1):
        row = test_df.iloc[idx]
        ptid = row["ptid"]
        actual_label = test_y[idx]
        actual_status = "Progressive MCI (pMCI) -> Diagnosed with AD within 36 months" if actual_label == 1 else "Stable MCI (sMCI) -> Remained stable throughout follow-up"

        img_tensor = test_img[idx:idx+1]
        cli_tensor = test_cli[idx:idx+1]

        with torch.no_grad():
            logit = model(img_tensor, cli_tensor)
            prob = float(torch.sigmoid(logit).item())
            pred_label = 1 if prob >= 0.50 else 0

        pred_status = "Progressive MCI (pMCI - High Risk)" if pred_label == 1 else "Stable MCI (sMCI - Low Risk)"
        is_correct = (pred_label == actual_label)

        print(f"\n[Participant Case #{case_num}]: ID: {ptid} | Sex: {row['sex']}")
        print(f"  • Baseline Demographics & Clinical Scores:")
        print(f"      - Age                     : {row['age_years_approx']:.0f} years")
        print(f"      - Education               : {row['education_years']:.0f} years")
        print(f"      - MMSE Score (out of 30)  : {row['mmse_score']:.1f}")
        print(f"      - ADAS-Cog 13 Total Score : {row['adas_score']:.2f}")
        print(f"  • Structural MRI Input:")
        print(f"      - Modality / Sequence     : Baseline T1-weighted MP-RAGE (3D Volume: {tuple(img_tensor.shape[2:])})")
        print(f"      - Preprocessing           : Canonical orientation, brain masking, intensity z-score")
        print(f"  • Multimodal Model Forward Pass:")
        print(f"      - 3D CNN Feature Vector   : 32 extracted spatial neuroanatomical features")
        print(f"      - Clinical Feature Vector : 16 extracted cognitive/demographic features")
        print(f"      - Fusion Layer            : 48-dimensional concatenated multimodal representation")
        print(f"  • Model Prediction:")
        print(f"      - Estimated Probability   : {prob * 100:.2f}% risk of AD conversion")
        print(f"      - Classification          : {pred_status}")
        print(f"  • Ground Truth (from 36m follow-up): {actual_status}")
        result_tag = "[PASS] CORRECT PREDICTION" if is_correct else "[CHECK] MISCLASSIFIED"
        print(f"  • Verification            : {result_tag}")

    print("\n" + "=" * 80)
    print("Summary of held-out test cohort ($n=23$):")
    with torch.no_grad():
        all_probs = torch.sigmoid(model(test_img, test_cli)).numpy()
        all_preds = (all_probs >= 0.50).astype(int)
    correct = (all_preds == test_y).sum()
    print(f"  • Correctly Classified    : {correct}/{len(test_y)} ({correct/len(test_y)*100:.1f}%)")
    tp = ((all_preds == 1) & (test_y == 1)).sum()
    fn = ((all_preds == 0) & (test_y == 1)).sum()
    print(f"  • Progressive MCI Sensitivity: {tp}/{tp+fn} ({tp/(tp+fn)*100:.1f}% detected)")
    print(f"  • False Negative Rate     : {fn}/{tp+fn} ({fn/(tp+fn)*100:.1f}%)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
