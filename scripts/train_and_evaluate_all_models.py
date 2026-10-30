"""Train and evaluate Clinical-only MLP, MRI-only 3D CNN, and Combined Multimodal Fusion Net.

Evaluates on an independent held-out test set with:
  - Accuracy, Sensitivity, Specificity, F1-score, ROC-AUC, False-Negative Rate
  - Sex-stratified evaluation (Female vs Male)
  - Exports training/validation curves across epochs for all models
  - Exports predictions and comparative tables
"""
from __future__ import annotations
import argparse
import json
import os
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "metadata" / "mri_training_manifest.csv"
CACHE_PATH = ROOT / "data" / "processed" / "mri_preprocessed_cache.pt"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

CLINICAL_COLS = ["age_years_approx", "education_years", "mmse_score", "adas_score", "sex_binary"]
VOLUME_SHAPE = (48, 56, 48)
SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preprocess_volume(path: Path, target_shape: tuple[int, int, int] = VOLUME_SHAPE) -> np.ndarray:
    """Canonicalize orientation, clip extreme intensities, normalize brain tissue, and resize."""
    image = nib.as_closest_canonical(nib.load(str(path)))
    volume = np.asarray(image.dataobj, dtype=np.float32)
    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)

    # Brain tissue masking (nonzero)
    nonzero = np.abs(volume) > 1e-4
    values = volume[nonzero]
    if values.size < 100:
        raise ValueError(f"Too few nonzero voxels in volume: {path.name}")

    # Robust intensity clipping
    low, high = np.percentile(values, [0.5, 99.5])
    volume = np.clip(volume, low, high)

    # Standardize intensity on brain tissue
    mean = float(volume[nonzero].mean())
    std = float(volume[nonzero].std())
    volume = (volume - mean) / max(std, 1e-5)

    # Interpolate to target shape
    tensor = torch.from_numpy(volume.copy())[None, None]
    tensor = F.interpolate(tensor, size=target_shape, mode="trilinear", align_corners=False)
    return tensor[0, 0].numpy().astype(np.float32)


def _process_one_volume(task: tuple[int, str]) -> tuple[int, np.ndarray]:
    idx, rel_path = task
    p = Path(rel_path)
    if not p.is_absolute():
        p = ROOT / p
    vol = preprocess_volume(p, VOLUME_SHAPE)
    return idx, vol


def get_cached_volumes(manifest_df: pd.DataFrame, force_recompute: bool = False) -> torch.Tensor:
    """Preprocess and cache MRI volumes to accelerate training with parallel multiprocessing."""
    if CACHE_PATH.is_file() and not force_recompute:
        try:
            cached_data = torch.load(CACHE_PATH, weights_only=True)
            if cached_data.shape[0] == len(manifest_df) and tuple(cached_data.shape[2:]) == VOLUME_SHAPE:
                print(f"Loaded {len(cached_data)} preprocessed volumes from cache: {CACHE_PATH.name}", flush=True)
                return cached_data
        except Exception:
            pass

    print(f"Preprocessing {len(manifest_df)} MRI volumes into resolution {VOLUME_SHAPE} in parallel...", flush=True)
    from concurrent.futures import ProcessPoolExecutor
    tasks = [(idx, str(row["nifti_path"])) for idx, row in manifest_df.iterrows()]
    results = [None] * len(tasks)

    num_workers = min(8, os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for idx, vol in executor.map(_process_one_volume, tasks):
            results[idx] = vol
            print(f"  Completed volume [{idx+1:03d}/{len(tasks):03d}]", flush=True)

    stacked = torch.from_numpy(np.stack(results)[:, None])
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stacked, CACHE_PATH)
    print(f"Saved preprocessed volume cache to: {CACHE_PATH}", flush=True)
    return stacked


# ==========================================
# MODEL DEFINITIONS
# ==========================================

class ClinicalMLP(nn.Module):
    """Clinical-only Multi-Layer Perceptron."""
    def __init__(self, in_features: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

    def forward(self, x_clin: torch.Tensor) -> torch.Tensor:
        return self.net(x_clin).squeeze(1)


class MRI3DCNN(nn.Module):
    """MRI-only 3D Convolutional Neural Network."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 48x56x48 -> 24x28x24
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2),
            # Block 2: 24x28x24 -> 12x14x12
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),
            # Block 3: 12x14x12 -> 6x7x6
            nn.Conv3d(32, 48, kernel_size=3, padding=1),
            nn.BatchNorm3d(48),
            nn.ReLU(),
            nn.MaxPool3d(2),
            # Global pooling: -> 1x1x1
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(48, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, x_img: torch.Tensor) -> torch.Tensor:
        feat = self.features(x_img)
        return self.classifier(feat).squeeze(1)


class MultimodalFusionNet(nn.Module):
    """Combined 3D CNN + Clinical MLP Multimodal Deep Learning Model."""
    def __init__(self, clinical_size: int = 5):
        super().__init__()
        # Image encoder branch
        self.image_encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 48, kernel_size=3, padding=1),
            nn.BatchNorm3d(48),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(48, 32),
            nn.ReLU(),
        )
        # Clinical encoder branch
        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_size, 24),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(24, 16),
            nn.ReLU(),
        )
        # Fusion classification head (32 + 16 = 48)
        self.classifier = nn.Sequential(
            nn.Linear(48, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

    def forward(self, x_img: torch.Tensor, x_clin: torch.Tensor) -> torch.Tensor:
        img_feat = self.image_encoder(x_img)
        clin_feat = self.clinical_encoder(x_clin)
        fused = torch.cat([img_feat, clin_feat], dim=1)
        return self.classifier(fused).squeeze(1)


# ==========================================
# METRICS & EVALUATION
# ==========================================

def calculate_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Calculate comprehensive evaluation metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    n_total = len(y_true)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())

    acc = float(accuracy_score(y_true, y_pred))
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    fnr = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0
    
    if len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score(y_true, y_prob))
    else:
        auc = None

    return {
        "n": n_total,
        "n_pMCI": n_pos,
        "n_sMCI": n_neg,
        "accuracy": round(acc, 4),
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "precision": round(prec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4) if auc is not None else None,
        "false_negative_rate": round(fnr, 4),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
    }


def train_single_model(model_name: str, model: nn.Module, train_loader: DataLoader,
                       val_data: tuple, device: torch.device, epochs: int = 35,
                       pos_weight: float = 1.0, lr: float = 1e-4) -> tuple[dict, list[dict]]:
    """Train one model while recording loss and accuracy across epochs."""
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    
    history = []
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        train_preds, train_targets = [], []

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            if model_name == "Clinical-only MLP":
                x_c, y_b = batch[0].to(device), batch[1].to(device)
                logits = model(x_c)
            elif model_name == "MRI-only 3D CNN":
                x_i, y_b = batch[0].to(device), batch[1].to(device)
                logits = model(x_i)
            else:  # Combined Multimodal
                x_i, x_c, y_b = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                logits = model(x_i, x_c)

            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
            train_targets.extend(y_b.cpu().numpy())

        train_loss = float(np.mean(train_losses))
        train_acc = float(accuracy_score(train_targets, (np.array(train_preds) >= 0.5).astype(int)))

        # Validation pass
        model.eval()
        with torch.no_grad():
            if model_name == "Clinical-only MLP":
                x_val, y_val = val_data[1].to(device), val_data[2].to(device)
                val_logits = model(x_val)
            elif model_name == "MRI-only 3D CNN":
                x_val, y_val = val_data[0].to(device), val_data[2].to(device)
                val_logits = model(x_val)
            else:
                x_img_val, x_cli_val, y_val = val_data[0].to(device), val_data[1].to(device), val_data[2].to(device)
                val_logits = model(x_img_val, x_cli_val)

            val_loss = float(criterion(val_logits, y_val).item())
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_targets = y_val.cpu().numpy()
            val_acc = float(accuracy_score(val_targets, (val_probs >= 0.5).astype(int)))

        history.append({
            "model": model_name,
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
        })

        if epoch in {1, 5, 10, 20, 30, epochs}:
            print(f"[{model_name}] Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}")

    return model, history


def main() -> None:
    set_seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Training manifest missing: {MANIFEST}")

    df = pd.read_csv(MANIFEST)
    df.columns = df.columns.str.strip().str.lower()
    
    # Filter confirmed and existing
    df = df[df["t1_confirmed"].astype("string").str.strip().str.casefold().isin({"yes", "y", "true", "1"})].copy()
    df["nifti_path"] = df["nifti_path"].astype("string").str.strip()
    df = df[df["nifti_path"].map(lambda p: (ROOT / p).is_file() if not Path(p).is_absolute() else Path(p).is_file())].copy()
    df = df.drop_duplicates(subset=["ptid"], keep="first").reset_index(drop=True)

    print(f"Loaded verified manifest with {len(df)} participants.")
    print("Outcome labels:", dict(df["label"].value_counts()))
    print("Sex counts:", dict(df["sex"].value_counts(dropna=False)))

    # Clinical features
    clinical_sub = df[["age_years_approx", "education_years", "mmse_score", "adas_score"]].apply(pd.to_numeric, errors="coerce")
    sex_binary = df["sex"].str.strip().str.casefold().map({"female": 0.0, "male": 1.0}).astype(float)
    clinical_sub["sex_binary"] = sex_binary

    y = df["label"].astype(int).to_numpy()
    indices = np.arange(len(df))

    # Stratified participant-level holdout split
    train_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=SEED, stratify=y)
    print(f"Split: Train={len(train_idx)} participants, Test={len(test_idx)} participants")

    # Strict normalization on train split only
    train_clin = clinical_sub.iloc[train_idx]
    medians = train_clin.median().fillna(0.0)
    means = train_clin.fillna(medians).mean()
    stds = train_clin.fillna(medians).std().replace(0, 1).fillna(1)
    
    x_clin_norm = ((clinical_sub.fillna(medians) - means) / stds).to_numpy(dtype=np.float32)
    x_clin_tensor = torch.from_numpy(x_clin_norm)

    # Load / Cache MRI volumes
    x_img_tensor = get_cached_volumes(df)
    y_tensor = torch.from_numpy(y.astype(np.float32))

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    # Class imbalance weighting
    pos_count = max(int((y[train_idx] == 1).sum()), 1)
    neg_count = max(int((y[train_idx] == 0).sum()), 1)
    pos_weight = neg_count / pos_count
    print(f"Class weights: neg={neg_count}, pos={pos_count}, pos_weight={pos_weight:.3f}")

    # Datasets and Loaders
    batch_size = 6
    train_loader_clin = DataLoader(TensorDataset(x_clin_tensor[train_idx], y_tensor[train_idx]), batch_size=batch_size, shuffle=True)
    train_loader_img = DataLoader(TensorDataset(x_img_tensor[train_idx], y_tensor[train_idx]), batch_size=batch_size, shuffle=True)
    train_loader_multi = DataLoader(TensorDataset(x_img_tensor[train_idx], x_clin_tensor[train_idx], y_tensor[train_idx]), batch_size=batch_size, shuffle=True)

    val_data = (x_img_tensor[test_idx], x_clin_tensor[test_idx], y_tensor[test_idx])

    # ==========================================
    # TRAIN ALL 3 MODELS
    # ==========================================
    epochs = 35
    all_histories = []
    model_predictions = {}
    evaluation_results = {}

    models_to_train = [
        ("Clinical-only MLP", ClinicalMLP(x_clin_norm.shape[1]).to(device), train_loader_clin, 2e-3),
        ("MRI-only 3D CNN", MRI3DCNN().to(device), train_loader_img, 1e-4),
        ("Combined CNN + MLP", MultimodalFusionNet(x_clin_norm.shape[1]).to(device), train_loader_multi, 1.5e-4),
    ]

    for model_name, model, loader, lr in models_to_train:
        print(f"\n==========================================")
        print(f"Training: {model_name}")
        print(f"==========================================")
        set_seed(SEED)
        trained_model, hist = train_single_model(
            model_name=model_name,
            model=model,
            train_loader=loader,
            val_data=val_data,
            device=device,
            epochs=epochs,
            pos_weight=pos_weight,
            lr=lr,
        )
        all_histories.extend(hist)

        # Held-out Test Predictions
        trained_model.eval()
        with torch.no_grad():
            if model_name == "Clinical-only MLP":
                prob = torch.sigmoid(trained_model(val_data[1].to(device))).cpu().numpy()
            elif model_name == "MRI-only 3D CNN":
                prob = torch.sigmoid(trained_model(val_data[0].to(device))).cpu().numpy()
            else:
                prob = torch.sigmoid(trained_model(val_data[0].to(device), val_data[1].to(device))).cpu().numpy()

        model_predictions[model_name] = prob
        y_test = y[test_idx]

        # Overall metrics
        overall_m = calculate_metrics(y_test, prob)
        
        # Sex-stratified metrics
        by_sex = {}
        for sex_label in ["Female", "Male"]:
            sex_mask = df.iloc[test_idx]["sex"].str.strip().str.casefold() == sex_label.casefold()
            if sex_mask.any():
                by_sex[sex_label] = calculate_metrics(y_test[sex_mask], prob[sex_mask])

        evaluation_results[model_name] = {
            "overall": overall_m,
            "by_sex": by_sex,
        }

        # Save model checkpoint
        ckpt_path = MODEL_DIR / f"{model_name.lower().replace(' ', '_').replace('+', 'and')}.pt"
        torch.save(trained_model.state_dict(), ckpt_path)
        print(f"Saved {model_name} checkpoint to {ckpt_path.name}")

    # ==========================================
    # SAVE EVALUATION ARTIFACTS
    # ==========================================
    history_df = pd.DataFrame(all_histories)
    history_path = REPORT_DIR / "training_validation_history.csv"
    history_df.to_csv(history_path, index=False)
    print(f"\nSaved training history to {history_path}")

    # Comparative Results Table
    table_rows = []
    for model_name, res in evaluation_results.items():
        om = res["overall"]
        table_rows.append({
            "Model": model_name,
            "Accuracy": om["accuracy"],
            "Sensitivity": om["sensitivity"],
            "Specificity": om["specificity"],
            "F1-score": om["f1"],
            "ROC-AUC": om["roc_auc"],
            "False-Negative Rate": om["false_negative_rate"],
        })
    comparison_df = pd.DataFrame(table_rows)
    comparison_path = REPORT_DIR / "model_comparison_table.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print("\n--- MODEL COMPARISON TABLE ---")
    print(comparison_df.to_string(index=False))

    # Comprehensive metrics JSON
    metrics_json_path = REPORT_DIR / "evaluation_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2)
    print(f"\nSaved detailed metrics JSON to {metrics_json_path}")

    # Predictions CSV
    pred_df = pd.DataFrame({
        "actual_label": y[test_idx],
        "sex": df.iloc[test_idx]["sex"].to_numpy(),
        "prob_clinical_mlp": model_predictions["Clinical-only MLP"],
        "pred_clinical_mlp": (model_predictions["Clinical-only MLP"] >= 0.5).astype(int),
        "prob_mri_3dcnn": model_predictions["MRI-only 3D CNN"],
        "pred_mri_3dcnn": (model_predictions["MRI-only 3D CNN"] >= 0.5).astype(int),
        "prob_combined": model_predictions["Combined CNN + MLP"],
        "pred_combined": (model_predictions["Combined CNN + MLP"] >= 0.5).astype(int),
    })
    # Also save standard test_predictions.csv compatible with matlab figure generator
    standard_pred_df = pd.DataFrame({
        "actual_label": y[test_idx],
        "predicted_probability": model_predictions["Combined CNN + MLP"],
        "predicted_label": (model_predictions["Combined CNN + MLP"] >= 0.5).astype(int),
        "sex": df.iloc[test_idx]["sex"].to_numpy(),
    })
    standard_pred_df.to_csv(REPORT_DIR / "test_predictions.csv", index=False)
    pred_df.to_csv(REPORT_DIR / "all_models_predictions.csv", index=False)
    print(f"Saved prediction records to {REPORT_DIR / 'test_predictions.csv'}")


if __name__ == "__main__":
    main()
