"""Train and visualize the multimodal pipeline on wholly synthetic data.

This is a software demonstration only. It does not load ADNI data, MRI files,
or participant identifiers. Its metrics and Grad-CAM image are not research
results and must never be presented as ADNI performance.

Run from the project folder:
  py -3.14 .\scripts\demo_synthetic.py
"""
from __future__ import annotations
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_multimodal import FusionNet, metrics

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
SEED = 23
N_PARTICIPANTS = 160
VOLUME_SHAPE = (32, 32, 32)


def make_synthetic_data():
    """Make toy brain-shaped volumes with an artificial class signal."""
    rng = np.random.default_rng(SEED)
    labels = np.tile(np.array([0, 1], dtype=np.int64), N_PARTICIPANTS // 2)
    rng.shuffle(labels)
    sex = np.tile(np.array([0, 1], dtype=np.int64), N_PARTICIPANTS // 2)
    rng.shuffle(sex)

    axis = np.arange(VOLUME_SHAPE[0], dtype=np.float32)
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")
    brain = (((x - 15.5) / 13.5) ** 2 + ((y - 15.5) / 13.5) ** 2 + ((z - 15.5) / 13.5) ** 2) < 1
    roi_left = np.exp(-(((x - 10) ** 2 + (y - 17) ** 2 + (z - 15) ** 2) / (2 * 3.0**2)))
    roi_right = np.exp(-(((x - 21) ** 2 + (y - 17) ** 2 + (z - 15) ** 2) / (2 * 3.0**2)))
    artificial_roi_signal = (roi_left + roi_right).astype(np.float32)
    volumes = []
    for label in labels:
        volume = rng.normal(0.0, 0.25, size=VOLUME_SHAPE).astype(np.float32)
        volume[brain] += 0.45
        # Toy class-dependent signal, deliberately placed in a known region.
        volume += float(label) * 0.8 * artificial_roi_signal
        volumes.append(volume)
    volumes = np.stack(volumes)[:, None]

    age = rng.normal(72, 5, N_PARTICIPANTS)
    education = rng.normal(15, 2.5, N_PARTICIPANTS)
    mmse = rng.normal(27 - 2.0 * labels, 1.5, N_PARTICIPANTS)
    adas = rng.normal(12 + 4.5 * labels, 3.0, N_PARTICIPANTS)
    clinical = np.column_stack([age, education, mmse, adas, sex]).astype(np.float32)
    return volumes, clinical, labels, sex, artificial_roi_signal


def gradcam_for_image(model, image, clinical, output_shape):
    """Compute a basic Grad-CAM from the last 3D convolution layer."""
    saved = {}
    target_layer = model.image_encoder[8]

    def save_activation(_module, _inputs, output):
        saved["activation"] = output
        output.register_hook(lambda gradient: saved.__setitem__("gradient", gradient))

    handle = target_layer.register_forward_hook(save_activation)
    model.zero_grad(set_to_none=True)
    score = model(image, clinical).sum()
    score.backward()
    handle.remove()
    activation = saved["activation"].detach()
    gradient = saved["gradient"].detach()
    weights = gradient.mean(dim=(2, 3, 4), keepdim=True)
    heatmap = torch.relu((weights * activation).sum(dim=1, keepdim=True))
    heatmap = nn.functional.interpolate(heatmap, size=output_shape, mode="trilinear", align_corners=False)
    heatmap = heatmap[0, 0].cpu().numpy()
    maximum = float(heatmap.max())
    return heatmap / maximum if maximum > 0 else heatmap


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    volumes, clinical, labels, sex, toy_roi = make_synthetic_data()
    indices = np.arange(N_PARTICIPANTS)
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=SEED, stratify=labels)

    # Fit clinical imputation/scaling on training data only, as in a real analysis.
    medians = np.nanmedian(clinical[train_idx], axis=0)
    means = np.mean(clinical[train_idx], axis=0)
    stds = np.std(clinical[train_idx], axis=0)
    stds[stds == 0] = 1
    clinical_scaled = ((clinical - medians) - (means - medians)) / stds

    x_image = torch.from_numpy(volumes)
    x_clinical = torch.from_numpy(clinical_scaled.astype(np.float32))
    y_tensor = torch.from_numpy(labels.astype(np.float32))
    train_set = TensorDataset(x_image[train_idx], x_clinical[train_idx], y_tensor[train_idx])
    loader = DataLoader(train_set, batch_size=16, shuffle=True)
    model = FusionNet(clinical_size=clinical.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    loss_history = []
    model.train()
    for epoch in range(1, 21):
        batch_losses = []
        for image_batch, clinical_batch, target_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(image_batch, clinical_batch), target_batch)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach()))
        loss_history.append(float(np.mean(batch_losses)))
        if epoch in {1, 5, 10, 15, 20}:
            print(f"Synthetic demo epoch {epoch:02d}/20 - loss: {loss_history[-1]:.4f}")

    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(x_image[test_idx], x_clinical[test_idx])).numpy()
    y_test = labels[test_idx]
    results = {
        "notice": "SYNTHETIC DEMO ONLY; NOT ADNI DATA OR ADNI PERFORMANCE",
        "n_synthetic_participants": int(N_PARTICIPANTS),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "overall": metrics(y_test, probabilities),
        "by_sex": {},
    }
    test_sex = sex[test_idx]
    for group_value, group_name in [(0, "Female (synthetic)"), (1, "Male (synthetic)")]:
        mask = test_sex == group_value
        results["by_sex"][group_name] = metrics(y_test[mask], probabilities[mask])

    # Select one test example and create a Grad-CAM overlay for the artificial signal.
    example_pos = int(np.argmax(probabilities))
    example_idx = int(test_idx[example_pos])
    heatmap = gradcam_for_image(
        model,
        x_image[example_idx:example_idx + 1],
        x_clinical[example_idx:example_idx + 1],
        VOLUME_SHAPE,
    )
    slice_index = VOLUME_SHAPE[0] // 2
    model.train(False)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "SYNTHETIC_demo_only.pt"
    report_path = REPORT_DIR / "SYNTHETIC_demo_metrics.json"
    figure_path = REPORT_DIR / "SYNTHETIC_demo_figure.png"
    torch.save({"model_state": model.state_dict(), "synthetic_only": True, "seed": SEED}, model_path)
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    axes[0].plot(range(1, len(loss_history) + 1), loss_history, color="#276FBF")
    axes[0].set(title="Toy training loss", xlabel="Epoch", ylabel="BCE loss")
    cm = np.array([
        [((y_test == 0) & (probabilities < 0.5)).sum(), ((y_test == 0) & (probabilities >= 0.5)).sum()],
        [((y_test == 1) & (probabilities < 0.5)).sum(), ((y_test == 1) & (probabilities >= 0.5)).sum()],
    ])
    axes[1].imshow(cm, cmap="Blues")
    axes[1].set(title="Toy holdout confusion matrix", xlabel="Predicted", ylabel="Actual",
                xticks=[0, 1], yticks=[0, 1])
    for (row, col), value in np.ndenumerate(cm):
        axes[1].text(col, row, str(value), ha="center", va="center")
    axes[2].imshow(volumes[example_idx, 0, slice_index], cmap="gray")
    axes[2].imshow(heatmap[slice_index], cmap="jet", alpha=0.45, vmin=0, vmax=1)
    axes[2].set(title="Toy Grad-CAM overlay", xticks=[], yticks=[])
    fig.suptitle("SYNTHETIC DEMONSTRATION — NOT ADNI DATA OR RESULTS", color="#A11", weight="bold")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("\nSYNTHETIC DEMONSTRATION ONLY — do not report these metrics as ADNI findings.")
    print(json.dumps(results, indent=2))
    print(f"\nSaved demo figure: {figure_path}")
    print(f"Saved demo metrics: {report_path}")
    print(f"Saved demo-only weights: {model_path}")


if __name__ == "__main__":
    main()
