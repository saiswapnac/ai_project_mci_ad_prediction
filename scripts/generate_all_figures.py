"""Generate publication-ready figures for the MCI-to-AD prediction project.

Generates:
  - Figure 1: Distribution of Classes (Section X: Fig. 1)
  - Figure 2: Data Preprocessing Output (Section X: Fig. 2)
  - Figure 3: Proposed Multimodal Model Architecture Diagram (Section X: Fig. 3)
  - Figure 4: Training and Validation Performance Curves (Section X: Fig. 4)
  - Figure 5: Held-Out ROC Curves (Clinical MLP vs MRI 3D CNN vs Multimodal)
  - Figure 6: Confusion Matrices for All 3 Models
  - Figure 7: Sex-Stratified Performance Comparison
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Set high-resolution aesthetics
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})


def plot_figure1_class_distribution(manifest_df: pd.DataFrame) -> None:
    """Figure 1: Distribution of Classes."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Overall class counts
    class_counts = manifest_df["label"].value_counts().sort_index()
    labels = ["Stable MCI (sMCI)", "Progressive MCI (pMCI)"]
    counts = [class_counts.get(0, 0), class_counts.get(1, 0)]
    total = sum(counts)
    pcts = [c / total * 100 for c in counts]
    colors = ["#2b5c8f", "#d95f02"]

    bars = axes[0].bar(labels, counts, color=colors, width=0.5, edgecolor="black", alpha=0.88)
    axes[0].set_title("Overall Cohort Target Distribution", fontweight="bold")
    axes[0].set_ylabel("Number of Participants")
    axes[0].set_ylim(0, max(counts) * 1.25)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    for bar, count, pct in zip(bars, counts, pcts):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2.0, height + 1.5,
                     f"{count}\n({pct:.1f}%)", ha="center", va="bottom", fontweight="bold")

    # Sex-stratified distribution
    grouped = manifest_df.groupby(["sex", "label"]).size().unstack(fill_value=0)
    sexes = ["Female", "Male"]
    x = np.arange(len(sexes))
    width = 0.35

    s_counts = [grouped.loc[s, 0] if s in grouped.index and 0 in grouped.columns else 0 for s in sexes]
    p_counts = [grouped.loc[s, 1] if s in grouped.index and 1 in grouped.columns else 0 for s in sexes]

    r1 = axes[1].bar(x - width/2, s_counts, width, label="sMCI (Stable)", color="#2b5c8f", edgecolor="black", alpha=0.88)
    r2 = axes[1].bar(x + width/2, p_counts, width, label="pMCI (Progressive)", color="#d95f02", edgecolor="black", alpha=0.88)

    axes[1].set_title("Class Distribution Stratified by Sex", fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"Female (n={sum(s_counts[0:1]+p_counts[0:1])})", f"Male (n={sum(s_counts[1:2]+p_counts[1:2])})"])
    axes[1].set_ylabel("Number of Participants")
    axes[1].set_ylim(0, max(max(s_counts), max(p_counts)) * 1.25)
    axes[1].legend(loc="upper right", framealpha=0.9)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    for bar, count in zip(r1, s_counts):
        axes[1].text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 1.0, f"{count}", ha="center", va="bottom")
    for bar, count in zip(r2, p_counts):
        axes[1].text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 1.0, f"{count}", ha="center", va="bottom")

    plt.tight_layout()
    out_file = FIGURE_DIR / "figure1_class_distribution.png"
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_figure2_data_preprocessing(manifest_df: pd.DataFrame, sample_nii_path: Path) -> None:
    """Figure 2: Data Preprocessing Output (Clinical + MRI)."""
    import nibabel as nib
    fig = plt.figure(figsize=(15, 6.5))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.1, 1.1, 1.2, 1.2], hspace=0.35, wspace=0.28)

    # 1. Age Distribution (Raw vs Standardized)
    ax1 = fig.add_subplot(gs[0, 0])
    raw_age = manifest_df["age_years_approx"].dropna()
    ax1.hist(raw_age, bins=12, color="#377eb8", edgecolor="black", alpha=0.75)
    ax1.set_title("Raw Age (Years)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Count")
    ax1.grid(axis="y", linestyle=":", alpha=0.6)

    ax2 = fig.add_subplot(gs[1, 0])
    norm_age = (raw_age - raw_age.mean()) / raw_age.std()
    ax2.hist(norm_age, bins=12, color="#4daf4a", edgecolor="black", alpha=0.75)
    ax2.set_title("Standardized Age (z-score)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Count")
    ax2.grid(axis="y", linestyle=":", alpha=0.6)

    # 2. ADAS-Cog Total (Raw vs Standardized)
    ax3 = fig.add_subplot(gs[0, 1])
    raw_adas = manifest_df["adas_score"].dropna()
    ax3.hist(raw_adas, bins=12, color="#984ea3", edgecolor="black", alpha=0.75)
    ax3.set_title("Raw ADAS-Cog 13", fontsize=11, fontweight="bold")
    ax3.grid(axis="y", linestyle=":", alpha=0.6)

    ax4 = fig.add_subplot(gs[1, 1])
    norm_adas = (raw_adas - raw_adas.mean()) / raw_adas.std()
    ax4.hist(norm_adas, bins=12, color="#ff7f00", edgecolor="black", alpha=0.75)
    ax4.set_title("Standardized ADAS-Cog (z-score)", fontsize=11, fontweight="bold")
    ax4.grid(axis="y", linestyle=":", alpha=0.6)

    # 3. MRI Preprocessing: Axial, Coronal, Sagittal Slices
    img = nib.as_closest_canonical(nib.load(str(sample_nii_path)))
    vol = np.asarray(img.dataobj, dtype=np.float32)
    # Normalize
    nonzero = np.abs(vol) > 1e-4
    if nonzero.any():
        p1, p99 = np.percentile(vol[nonzero], [1, 99])
        vol = np.clip(vol, p1, p99)
        vol = (vol - vol[nonzero].mean()) / max(vol[nonzero].std(), 1e-5)

    cx, cy, cz = [s // 2 for s in vol.shape]

    ax_axial = fig.add_subplot(gs[0, 2])
    ax_axial.imshow(np.rot90(vol[:, :, cz]), cmap="bone")
    ax_axial.set_title(f"Axial Slice (z={cz})", fontsize=11, fontweight="bold")
    ax_axial.axis("off")

    ax_coronal = fig.add_subplot(gs[1, 2])
    ax_coronal.imshow(np.rot90(vol[:, cy, :]), cmap="bone")
    ax_coronal.set_title(f"Coronal Slice (y={cy})", fontsize=11, fontweight="bold")
    ax_coronal.axis("off")

    ax_sagittal = fig.add_subplot(gs[:, 3])
    ax_sagittal.imshow(np.rot90(vol[cx, :, :]), cmap="bone")
    ax_sagittal.set_title(f"Sagittal Slice (x={cx})\n(Canonical T1 MP-RAGE)", fontsize=11, fontweight="bold")
    ax_sagittal.axis("off")

    fig.suptitle("Data Preprocessing Output: Standardized Clinical Tabular Features & Preprocessed T1 MRI Scans",
                 fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    out_file = FIGURE_DIR / "figure2_data_preprocessing.png"
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_figure3_multimodal_architecture() -> None:
    """Figure 3: Proposed Multimodal Model Architecture."""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def draw_box(x, y, w, h, title, subtitle, color, edgecolor="#1f4e79"):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=1.2",
                                      facecolor=color, edgecolor=edgecolor, linewidth=1.8)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.62, title, ha="center", va="center", fontsize=11, fontweight="bold", color="#0f2537")
        ax.text(x + w/2, y + h*0.28, subtitle, ha="center", va="center", fontsize=9.5, color="#2c3e50")

    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=2.0, color="#1f4e79", mutation_scale=15))
        if label:
            ax.text((x1 + x2)/2, (y1 + y2)/2 + 2, label, ha="center", va="center", fontsize=9, color="#1f4e79", fontweight="bold")

    # Image Branch (Top)
    draw_box(4, 65, 18, 22, "Baseline T1 MRI", "Shape: (1, 48, 56, 48)\nStandardized Intensity", "#d9edf7")
    draw_box(28, 65, 20, 22, "3D CNN Encoder", "3x Conv3D + BatchNorm\nMaxPool3D (x3) + AvgPool", "#c4e3f3")
    draw_box(54, 69, 14, 14, "Image Vector", "32 Features", "#a6d5ec")

    draw_arrow(22, 76, 28, 76)
    draw_arrow(48, 76, 54, 76)

    # Clinical Branch (Bottom)
    draw_box(4, 15, 18, 22, "Clinical Data", "Age, Sex, Education\nMMSE, ADAS-Cog", "#fcf8e3")
    draw_box(28, 15, 20, 22, "Clinical MLP", "Dense(24) -> BN -> ReLU\nDropout -> Dense(16)", "#fef1b8")
    draw_box(54, 19, 14, 14, "Clinical Vector", "16 Features", "#fde891")

    draw_arrow(22, 26, 28, 26)
    draw_arrow(48, 26, 54, 26)

    # Multimodal Fusion (Center)
    draw_box(73, 40, 10, 22, "Feature\nConcatenation", "48-dim Vector\n[32 Image + 16 Clin]", "#dff0d8", "#3c763d")
    
    # Arrows to Fusion
    ax.annotate("", xy=(73, 56), xytext=(68, 76),
                arrowprops=dict(arrowstyle="-|>", lw=2.0, color="#1f4e79", mutation_scale=15))
    ax.annotate("", xy=(73, 46), xytext=(68, 26),
                arrowprops=dict(arrowstyle="-|>", lw=2.0, color="#1f4e79", mutation_scale=15))

    # Classification Head (Right)
    draw_box(86, 39, 12, 24, "Classification Head", "Dense(32) -> ReLU\nDense(16) -> Logits\nSigmoid Activation", "#f2dede", "#a94442")
    draw_arrow(83, 51, 86, 51)

    # Final Output
    ax.text(92, 22, "Output Prediction:\nP(pMCI) Progression\nwithin 36 Months",
            ha="center", va="center", fontsize=10.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#fff", edgecolor="#a94442", lw=1.5))
    ax.annotate("", xy=(92, 30), xytext=(92, 39),
                arrowprops=dict(arrowstyle="-|>", lw=2.0, color="#a94442", mutation_scale=15))

    ax.set_title("Proposed Multimodal 3D CNN + Clinical MLP Deep Learning Architecture",
                 fontsize=14, fontweight="bold", y=0.98)

    plt.tight_layout()
    out_file = FIGURE_DIR / "figure3_multimodal_architecture.png"
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_figure4_training_validation(history_df: pd.DataFrame) -> None:
    """Figure 4: Training and Validation Performance (Loss & Accuracy across Epochs)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    models = history_df["model"].unique()
    colors = {"Clinical-only MLP": "#377eb8", "MRI-only 3D CNN": "#ff7f00", "Combined CNN + MLP": "#4daf4a"}

    # Left: Training and Validation Loss
    for m in models:
        sub = history_df[history_df["model"] == m]
        c = colors.get(m, "#333")
        axes[0].plot(sub["epoch"], sub["train_loss"], label=f"{m} (Train)", color=c, lw=2.0, linestyle="-")
        axes[0].plot(sub["epoch"], sub["val_loss"], label=f"{m} (Val)", color=c, lw=1.8, linestyle="--")

    axes[0].set_title("Training and Validation Loss Curves", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Cross-Entropy Loss")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="upper right", framealpha=0.9, fontsize=9)

    # Right: Training and Validation Accuracy
    for m in models:
        sub = history_df[history_df["model"] == m]
        c = colors.get(m, "#333")
        axes[1].plot(sub["epoch"], sub["train_acc"], label=f"{m} (Train)", color=c, lw=2.0, linestyle="-")
        axes[1].plot(sub["epoch"], sub["val_acc"], label=f"{m} (Val)", color=c, lw=1.8, linestyle="--")

    axes[1].set_title("Training and Validation Accuracy Curves", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.40, 1.02)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="lower right", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    out_file = FIGURE_DIR / "figure4_training_validation_curves.png"
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_figure5_roc_curves(pred_df: pd.DataFrame) -> None:
    """Figure 5: Held-out ROC curves for all 3 models."""
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    y_true = pred_df["actual_label"].to_numpy()

    model_configs = [
        ("Clinical-only MLP", pred_df["prob_clinical_mlp"].to_numpy(), "#377eb8", "-"),
        ("MRI-only 3D CNN", pred_df["prob_mri_3dcnn"].to_numpy(), "#ff7f00", "-."),
        ("Combined CNN + MLP", pred_df["prob_combined"].to_numpy(), "#4daf4a", "-"),
    ]

    for name, prob, color, ls in model_configs:
        fpr, tpr, _ = roc_curve(y_true, prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2.4, linestyle=ls, label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("Held-Out ROC Curves (36-Month Progression)", fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.92, fontsize=10.5)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_aspect("equal")

    plt.tight_layout()
    out_file = FIGURE_DIR / "figure5_heldout_roc_curves.png"
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_figure6_confusion_matrices(pred_df: pd.DataFrame) -> None:
    """Figure 6: Confusion matrices for all 3 models."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    y_true = pred_df["actual_label"].to_numpy()

    models = [
        ("Clinical-only MLP", pred_df["pred_clinical_mlp"].to_numpy()),
        ("MRI-only 3D CNN", pred_df["pred_mri_3dcnn"].to_numpy()),
        ("Combined CNN + MLP", pred_df["pred_combined"].to_numpy()),
    ]

    for ax, (name, y_pred) in zip(axes, models):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        cax = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(name, fontweight="bold", fontsize=11.5)
        tick_marks = np.arange(2)
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(["sMCI (0)", "pMCI (1)"])
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(["sMCI (0)", "pMCI (1)"])
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                pct = cm[i, j] / len(y_true) * 100
                ax.text(j, i, f"{cm[i, j]}\n({pct:.1f}%)",
                        ha="center", va="center", fontsize=12, fontweight="bold",
                        color="white" if cm[i, j] > thresh else "black")

    fig.suptitle("Confusion Matrices on Held-Out Test Set (Threshold = 0.50)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_file = FIGURE_DIR / "figure6_confusion_matrices.png"
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_figure7_sex_stratified(metrics_json_path: Path) -> None:
    """Figure 7: Sex-stratified metrics comparison."""
    with open(metrics_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Focus on the Combined multimodal model
    multi_sex = data.get("Combined CNN + MLP", {}).get("by_sex", {})
    if not multi_sex:
        print("No sex-stratified data available in JSON.")
        return

    fem = multi_sex.get("Female", {})
    mal = multi_sex.get("Male", {})

    metrics_keys = ["roc_auc", "sensitivity", "specificity", "f1", "false_negative_rate"]
    labels = ["ROC-AUC", "Sensitivity", "Specificity", "F1-Score", "False-Neg Rate"]

    fem_vals = [fem.get(k, 0.0) or 0.0 for k in metrics_keys]
    mal_vals = [mal.get(k, 0.0) or 0.0 for k in metrics_keys]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.5))
    r1 = ax.bar(x - width/2, fem_vals, width, label=f"Female (n={fem.get('n', 0)})", color="#984ea3", edgecolor="black", alpha=0.88)
    r2 = ax.bar(x + width/2, mal_vals, width, label=f"Male (n={mal.get('n', 0)})", color="#377eb8", edgecolor="black", alpha=0.88)

    ax.set_title("Combined CNN + MLP: Performance Stratified by Sex", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_ylabel("Score / Rate")
    ax.set_ylim(0.0, 1.15)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    for bar in r1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=9.5)
    for bar in r2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=9.5)

    plt.tight_layout()
    out_file = FIGURE_DIR / "figure7_sex_stratified_performance.png"
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def main() -> None:
    manifest_path = ROOT / "data" / "metadata" / "mri_training_manifest.csv"
    history_path = REPORT_DIR / "training_validation_history.csv"
    pred_path = REPORT_DIR / "all_models_predictions.csv"
    metrics_path = REPORT_DIR / "evaluation_metrics.json"

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest missing: {manifest_path}")

    manifest_df = pd.read_csv(manifest_path)
    manifest_df.columns = manifest_df.columns.str.strip().str.lower()
    confirmed = manifest_df[manifest_df["t1_confirmed"].astype("string").str.strip().str.casefold().isin({"yes", "y", "true", "1"})]

    print("Generating Figure 1: Distribution of Classes...")
    plot_figure1_class_distribution(confirmed)

    print("Generating Figure 2: Data Preprocessing Output...")
    sample_nii = ROOT / Path(confirmed.iloc[0]["nifti_path"])
    plot_figure2_data_preprocessing(confirmed, sample_nii)

    print("Generating Figure 3: Proposed Multimodal Model Architecture...")
    plot_figure3_multimodal_architecture()

    if history_path.is_file():
        print("Generating Figure 4: Training and Validation Performance Curves...")
        plot_figure4_training_validation(pd.read_csv(history_path))

    if pred_path.is_file():
        print("Generating Figure 5: Held-Out ROC Curves...")
        plot_figure5_roc_curves(pd.read_csv(pred_path))

        print("Generating Figure 6: Confusion Matrices...")
        plot_figure6_confusion_matrices(pd.read_csv(pred_path))

    if metrics_path.is_file():
        print("Generating Figure 7: Sex-Stratified Metrics Comparison...")
        plot_figure7_sex_stratified(metrics_path)

    print("\nAll publication figures successfully created in:", FIGURE_DIR)


if __name__ == "__main__":
    main()
