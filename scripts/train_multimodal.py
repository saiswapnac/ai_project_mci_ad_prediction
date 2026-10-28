"""Train an exploratory 3D CNN + clinical MLP on reviewed baseline scans.

Expected inputs:
  data/metadata/mri_training_manifest.csv (created by make_mri_manifest.py)
  one preprocessed NIfTI path per participant, with T1_CONFIRMED=yes

Run from the project folder:
  py -3.14 .\scripts\train_multimodal.py

The script uses a participant-level stratified holdout split. Results are a
prototype, not a reliable clinical or fairness assessment, especially with a
small MRI sample. Participant-level outputs stay local.
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "metadata" / "mri_training_manifest.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
CLINICAL = ["age_years_approx", "education_years", "mmse_score", "adas_score", "sex_binary"]
VOLUME_SHAPE = (64, 80, 64)
SEED = 42


def load_volume(path: Path) -> np.ndarray:
    """Canonicalize, robustly clip/z-score, and resize a NIfTI volume."""
    image = nib.as_closest_canonical(nib.load(str(path)))
    volume = np.asarray(image.dataobj, dtype=np.float32)
    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
    nonzero = np.abs(volume) > 1e-6
    values = volume[nonzero]
    if values.size < 100:
        raise ValueError(f"Image has too few nonzero voxels: {path.name}")
    low, high = np.percentile(values, [0.5, 99.5])
    volume = np.clip(volume, low, high)
    mask = np.abs(volume) > 1e-6
    mean = float(volume[mask].mean())
    std = float(volume[mask].std())
    volume = (volume - mean) / max(std, 1e-6)
    tensor = torch.from_numpy(volume.copy())[None, None]
    tensor = F.interpolate(tensor, size=VOLUME_SHAPE, mode="trilinear", align_corners=False)
    return tensor[0, 0].numpy().astype(np.float32)


class FusionNet(nn.Module):
    def __init__(self, clinical_size: int):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1), nn.BatchNorm3d(8), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(8, 16, 3, padding=1), nn.BatchNorm3d(16), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(),
            nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(32, 32), nn.ReLU(),
        )
        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_size, 16), nn.ReLU(), nn.Dropout(0.2), nn.Linear(16, 16), nn.ReLU()
        )
        self.classifier = nn.Sequential(nn.Linear(48, 16), nn.ReLU(), nn.Dropout(0.3), nn.Linear(16, 1))

    def forward(self, image: torch.Tensor, clinical: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.cat([self.image_encoder(image), self.clinical_encoder(clinical)], dim=1)).squeeze(1)


def metrics(y_true: np.ndarray, probability: np.ndarray) -> dict:
    prediction = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "auc": float(roc_auc_score(y_true, probability)) if len(np.unique(y_true)) == 2 else None,
        "accuracy": float(accuracy_score(y_true, prediction)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST, help="Local MRI training manifest")
    parser.add_argument("--model-output", type=Path, default=MODEL_DIR / "multimodal_prototype.pt")
    parser.add_argument("--report-output", type=Path, default=REPORT_DIR / "multimodal_prototype_metrics.json")
    args = parser.parse_args()
    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    checkpoint = args.model_output if args.model_output.is_absolute() else ROOT / args.model_output
    report_path = args.report_output if args.report_output.is_absolute() else ROOT / args.report_output
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if not manifest_path.exists():
        raise FileNotFoundError("Run scripts/make_mri_manifest.py and fill in its local scan fields first.")
    df = pd.read_csv(manifest_path, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    required = {"ptid", "label", "sex", "nifti_path", "t1_confirmed", *CLINICAL[:-1]}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Manifest missing required columns: {', '.join(missing)}")
    confirmed = df["t1_confirmed"].astype("string").str.strip().str.casefold().isin({"yes", "y", "true", "1"})
    df = df[confirmed].copy()
    df["nifti_path"] = df["nifti_path"].astype("string").str.strip()
    df = df[df["nifti_path"].ne("") & df["nifti_path"].notna()].copy()
    df["nifti_path"] = df["nifti_path"].map(lambda p: str((ROOT / p).resolve()) if not Path(p).is_absolute() else p)
    df = df[df["nifti_path"].map(lambda p: Path(p).is_file())].copy()
    if df["ptid"].duplicated().any():
        raise ValueError("Manifest must have exactly one selected baseline scan per participant (duplicate PTID found).")
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin([0, 1])].copy()
    df["sex"] = df["sex"].astype("string").str.strip().str.casefold().map({"female": "Female", "male": "Male"})
    df = df[df["sex"].notna()].copy()
    if len(df) < 20 or df["label"].nunique() < 2 or df["label"].value_counts().min() < 2:
        raise ValueError(f"Only {len(df)} eligible participants remain. The script has a 20-participant technical minimum and needs at least 2 per outcome; this is not a scientifically adequate sample-size threshold. Obtain more reviewed T1 scans.")

    clinical = df[["age_years_approx", "education_years", "mmse_score", "adas_score"]].apply(pd.to_numeric, errors="coerce")
    clinical["sex_binary"] = df["sex"].map({"Female": 0.0, "Male": 1.0}).astype(float)
    y = df["label"].astype(int).to_numpy()
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=SEED, stratify=y)
    train_values = clinical.iloc[train_idx]
    medians = train_values.median().fillna(0.0)
    means = train_values.fillna(medians).mean()
    stds = train_values.fillna(medians).std().replace(0, 1).fillna(1)
    x_clin = ((clinical.fillna(medians) - means) / stds).to_numpy(dtype=np.float32)

    print(f"Loading {len(df)} reviewed NIfTI scans locally...")
    images = np.stack([load_volume(Path(p)) for p in df["nifti_path"]])[:, None]
    x_img = torch.from_numpy(images)
    x_cli = torch.from_numpy(x_clin)
    y_tensor = torch.from_numpy(y.astype(np.float32))
    train_ds = TensorDataset(x_img[train_idx], x_cli[train_idx], y_tensor[train_idx])
    loader = DataLoader(train_ds, batch_size=min(4, len(train_ds)), shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionNet(len(CLINICAL)).to(device)
    pos = max(int((y[train_idx] == 1).sum()), 1)
    neg = max(int((y[train_idx] == 0).sum()), 1)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

    model.train()
    for epoch in range(1, 31):
        losses = []
        for image_batch, clinical_batch, target_batch in loader:
            image_batch, clinical_batch, target_batch = image_batch.to(device), clinical_batch.to(device), target_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(image_batch, clinical_batch), target_batch)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch in {1, 5, 10, 20, 30}:
            print(f"Epoch {epoch:02d}/30 - train loss: {np.mean(losses):.4f}")

    model.eval()
    with torch.no_grad():
        probability = torch.sigmoid(model(x_img[test_idx].to(device), x_cli[test_idx].to(device))).cpu().numpy()
    y_test = y[test_idx]
    results = {"overall": metrics(y_test, probability), "by_sex": {}}
    for sex in ["Female", "Male"]:
        local = df.iloc[test_idx]["sex"].to_numpy() == sex
        if local.any():
            results["by_sex"][sex] = metrics(y_test[local], probability[local])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "clinical_features": CLINICAL,
                "clinical_medians": medians.to_dict(), "clinical_means": means.to_dict(),
                "clinical_stds": stds.to_dict(), "volume_shape": VOLUME_SHAPE,
                "seed": SEED, "threshold": 0.5}, checkpoint)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nExploratory holdout metrics (small-sample estimates; no confidence intervals):")
    print(json.dumps(results, indent=2))
    print(f"\nSaved model locally: {checkpoint}")
    print(f"Saved aggregate metrics locally: {report_path}")
    print("This prototype resizes and normalizes scans but does not perform skull stripping, registration, or ROI extraction.")


if __name__ == "__main__":
    main()
