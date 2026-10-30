# Predicting MCI-to-Alzheimer’s Progression (36 Months) Using Multimodal Deep Learning

This project implements an end-to-end multimodal machine learning and deep learning pipeline to predict whether participants diagnosed with mild cognitive impairment (MCI) at baseline will convert to Alzheimer’s disease (AD) within 36 months (pMCI) or remain stable (sMCI), using baseline structural T1-weighted MRI scans and clinical/cognitive assessments from the Alzheimer’s Disease Neuroimaging Initiative (ADNI). In addition, it investigates sex-based performance differences by evaluating metrics separately for female and male subgroups.

---

## Project Status: Fully Implemented & Functioning

- **Clinical Cohort**: 240 baseline MCI participants derived from ADNI `DXSUM`, `PTDEMOG`, `MMSE`, and `ADAS` assessment tables, longitudinally tracked with strict 36-month diagnostic rules (147 pMCI, 93 sMCI; 0 missing sex, 0 missing age, 0 missing education, 0 missing MMSE).
- **Multimodal Verified Cohort**: 113 unique baseline MCI participants with verified T1 MP-RAGE structural MRI scans converted from DICOM to NIfTI format (61 pMCI, 52 sMCI; 77 Male, 36 Female).
- **Trained Deep Learning Models**:
  1. **Clinical-only MLP**: Multi-layer perceptron trained on baseline age, education, MMSE, ADAS-Cog 13, and sex.
  2. **MRI-only 3D CNN**: 3D Convolutional Neural Network extracting spatial neuroanatomical features from preprocessed T1 MRI volumes.
  3. **Combined CNN + MLP Multimodal Model**: Joint feature fusion network integrating 3D MRI spatial representations with clinical measurements.
- **Evaluation & Subgroup Analysis**: Evaluated on an independent participant-level held-out test set (stratified by outcome and sex). Evaluated across Accuracy, Sensitivity, Specificity, F1-score, ROC-AUC, and False-Negative Rate (overall and by sex).
- **Publication Figures**: All 4 report figures + 3 graph-based evaluation figures generated at 300 DPI in `reports/figures/`.

---

## Pipeline Overview (Phases 1–6)

```
[ ADNI Clinical Tables ]       [ Baseline T1 DICOM Scans ]
 (DXSUM, PTDEMOG, MMSE, ADAS)         (29,000+ DICOM slices)
             │                                   │
   Phase 1 & 2: Cohort & Outcomes          Phase 3 & 4: MRI Matching & Conversion
   (pMCI vs sMCI at 36m)                  (dcm2niix, Canonical, Intensity Z-score)
             │                                   │
             ▼                                   ▼
 [ Baseline Clinical Features ]          [ Standardized 3D MRI Volumes ]
 (Age, Educ, MMSE, ADAS, Sex)               (Shape: 48 x 56 x 48)
             │                                   │
             ├─────────────────┬─────────────────┤
             ▼                 ▼                 ▼
     Clinical-only MLP    MRI-only 3D CNN   Combined Multimodal Network
             │                 │                 │
             └─────────────────┼─────────────────┘
                               ▼
               Phase 6: Held-Out Evaluation
         - Overall Metrics (AUC, Sens, Spec, F1, FNR)
         - Sex-Stratified Subgroup Analysis (Female vs Male)
         - ROC Curves, Confusion Matrices, Loss Curves
```

1. **Phase 1: Data preparation**: ADNI assessment tables (`DXSUM`, `PTDEMOG`, `MMSE`, `ADAS`) converted to CSVs, cleaned, and unified.
2. **Phase 2: Outcome labeling**: 36-month diagnostic tracking. A participant is labeled **pMCI** if AD is documented within 36 months; **sMCI** if MCI is maintained at/after 36 months without AD conversion. Exclusions handle insufficient follow-up or diagnostic reversions.
3. **Phase 3: MRI matching**: Metadata matching baseline participant IDs and scan dates within a predefined window.
4. **Phase 4: MRI preprocessing**: Converted using `dcm2niix` to NIfTI; reoriented to canonical orientation; intensity clipped at 0.5%–99.5%; brain tissue z-score normalized; resized to $(48, 56, 48)$ using 3D trilinear interpolation.
5. **Phase 5: Clinical preprocessing**: Missing value median imputation and standardization ($z$-scores) learned strictly on the training partition to prevent leakage.
6. **Phase 6: Model training and evaluation**: Stratified holdout split; class-weighted BCE loss to manage class imbalance; overall and sex-stratified evaluation.

---

## Quick Start (Workflow Helper)

All steps can be executed locally from PowerShell using the built-in workflow helper `scripts/project_agent.py`:

```powershell
# 1. Check workspace readiness and status
py -3.14 .\scripts\project_agent.py status

# 2. Build or update the clinical cohort from ADNI tables
py -3.14 .\scripts\project_agent.py cohort

# 3. Convert matched DICOM series into NIfTI format
py -3.14 .\scripts\project_agent.py convert

# 4. Train and evaluate all 3 models (Clinical MLP, MRI 3D CNN, Combined)
py -3.14 .\scripts\project_agent.py train-all

# 5. Generate all publication figures (Figures 1 to 7)
py -3.14 .\scripts\project_agent.py figures

# 6. Display held-out evaluation and sex-stratified metrics
py -3.14 .\scripts\project_agent.py evaluate
```

---

## Experimental Evaluation Results

Results on the independent held-out test partition ($n = 23$; 12 pMCI, 11 sMCI; 10 Female, 13 Male):

### Model Comparison Table

| Model | Accuracy | Sensitivity (Recall) | Specificity | F1-Score | ROC-AUC | False-Negative Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clinical-only MLP** | **0.7826** | 0.7500 | **0.8182** | **0.7826** | **0.8864** | 0.2500 |
| **MRI-only 3D CNN** | 0.5217 | 0.3333 | 0.7273 | 0.4211 | 0.5606 | 0.6667 |
| **Combined CNN + MLP** | 0.6522 | **0.8333** | 0.4545 | 0.7143 | 0.7121 | **0.1667** |

> **Key Clinical Finding**: The multimodal **Combined CNN + MLP** model achieved the highest sensitivity (**0.8333**) and lowest false-negative rate (**0.1667** / 16.7%), successfully identifying 10 out of 12 progressing patients at baseline. In early Alzheimer's screening, minimizing false negatives is critical so high-risk individuals can access clinical trials and therapeutic interventions before irreversible neurodegeneration occurs.

---

### Sex-Stratified Performance Analysis

To evaluate whether the models perform differently across sex groups:

| Model | Subgroup | $n$ | Accuracy | Sensitivity | Specificity | F1-Score | ROC-AUC | False-Negative Rate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Combined CNN + MLP** | **Female** | 10 | 0.6000 | 0.8000 | 0.4000 | 0.6667 | 0.5600 | 0.2000 |
| **Combined CNN + MLP** | **Male** | 13 | 0.6923 | 0.8571 | 0.5000 | 0.7500 | 0.7619 | 0.1429 |
| **Clinical-only MLP** | **Female** | 10 | 0.9000 | 0.8000 | 1.0000 | 0.8889 | 0.9600 | 0.2000 |
| **Clinical-only MLP** | **Male** | 13 | 0.6923 | 0.7143 | 0.6667 | 0.7143 | 0.8095 | 0.2857 |
| **MRI-only 3D CNN** | **Female** | 10 | 0.6000 | 0.6000 | 0.6000 | 0.6000 | 0.4800 | 0.4000 |
| **MRI-only 3D CNN** | **Male** | 13 | 0.4615 | 0.1429 | 0.8333 | 0.2222 | 0.6429 | 0.8571 |

---

## Report Figures (`reports/figures/`)

All figures specified in Section X of the project specification have been generated and saved:

1. **Figure 1: Distribution of Classes** (`reports/figures/figure1_class_distribution.png`): Bar charts visualizing overall sMCI vs pMCI frequencies and the sex-stratified distribution illustrating class imbalance.
2. **Figure 2: Data Preprocessing Output** (`reports/figures/figure2_data_preprocessing.png`): Side-by-side display of standardized continuous clinical features (Age, ADAS-Cog) alongside orthogonal MRI cross-sections (Axial, Coronal, Sagittal).
3. **Figure 3: Proposed Multimodal Model Architecture** (`reports/figures/figure3_multimodal_architecture.png`): Visual schematic showing the 3D CNN feature extractor, Clinical MLP branch, feature concatenation fusion layer, and classification head.
4. **Figure 4: Training and Validation Performance** (`reports/figures/figure4_training_validation_curves.png`): Successive epoch curves tracking training and validation loss (binary cross-entropy) and accuracy across all models.
5. **Figure 5: Held-Out ROC Curves** (`reports/figures/figure5_heldout_roc_curves.png`): Receiver operating characteristic curves for Clinical MLP, MRI 3D CNN, and Combined Model.
6. **Figure 6: Confusion Matrices** (`reports/figures/figure6_confusion_matrices.png`): Confusion matrices for all models on the held-out test partition at threshold 0.50.
7. **Figure 7: Sex-Stratified Metrics Comparison** (`reports/figures/figure7_sex_stratified_performance.png`): Grouped bar chart comparing female and male performance across all primary metrics.

---

## Workspace Structure

```
mci_ad_prediction/
├── data/
│   ├── raw/                           # Raw ADNI CSV tables (DXSUM, PTDEMOG, MMSE, ADAS)
│   ├── metadata/                      # Labeled cohorts & MRI manifest
│   │   ├── master_cohort_clinical_only.csv  (240 clinical MCI participants)
│   │   └── mri_training_manifest.csv        (113 matched & verified scans)
│   └── processed/
│       ├── dicom_batch1/              # Original ADNI DICOM series
│       ├── dicom_batch2/              # Additional ADNI DICOM series
│       ├── nifti/                     # Converted uncompressed 3D NIfTI scans
│       └── mri_preprocessed_cache.pt  # Fast tensor cache (48x56x48)
├── dcm2niix_win/                      # dcm2niix Windows binary
├── models/                            # Trained PyTorch model checkpoints
│   ├── clinical-only_mlp.pt
│   ├── mri-only_3d_cnn.pt
│   └── combined_cnn_and_mlp.pt
├── reports/
│   ├── figures/                       # Publication-grade PNG figures (Figures 1-7)
│   ├── model_comparison_table.csv     # Comparative evaluation table
│   ├── evaluation_metrics.json        # Comprehensive metrics & subgroup breakdowns
│   ├── test_predictions.csv           # Held-out predictions
│   └── training_validation_history.csv# Loss and accuracy logs across epochs
└── scripts/
    ├── project_agent.py               # Main CLI workflow runner
    ├── build_cohort_from_tables.py    # Phase 1 & 2: Longitudinal cohort builder
    ├── match_mri_metadata.py          # Phase 3: Metadata matching
    ├── convert_all_manifest_dicoms.py # Phase 4: Batch DICOM-to-NIfTI converter
    ├── train_and_evaluate_all_models.py # Phase 5 & 6: Preprocessing, training & evaluation
    └── generate_all_figures.py        # Figure generator for Section X
```
