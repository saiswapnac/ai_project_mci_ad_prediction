# Predicting MCI-to-Alzheimer’s Progression

This ADNI research project aims to predict which participants with mild cognitive impairment (MCI) progress to Alzheimer’s disease within 36 months, using clinical data and baseline T1-weighted MRI. It will also compare model performance by sex.

## Current status

The clinical cohort is built from the ADNI `DXSUM`, `PTDEMOG`, `MMSE`, and `ADAS` tables. MRI matching is still in progress; the current metadata match is too small for MRI model training.

## Rebuild the clinical cohort

From this folder, run:

```powershell
py -3.14 .\scripts\build_cohort_from_tables.py
```

The script saves `data/metadata/master_cohort_clinical_only.csv`.

## MRI starter pipeline

Install the Python packages once:

```powershell
py -3.14 -m pip install -r .\requirements-modeling.txt
```

### Demonstrate the workflow before MRI arrives

Run the synthetic-data demo:

```powershell
py -3.14 .\scripts\demo_synthetic.py
```

It trains the fusion network on generated toy volumes and clinical values, then saves a figure with a loss curve, confusion matrix, and toy Grad-CAM overlay under `reports/`. The metrics, figure, and model weights are explicitly marked synthetic. They demonstrate code flow only and must not be described as ADNI results.

After downloading and reviewing baseline T1 scans in IDA:

1. Run `py -3.14 .\scripts\match_mri_metadata.py` to match the metadata locally.
2. Run `py -3.14 .\scripts\make_mri_manifest.py`; fill `NIFTI_PATH` and set `T1_CONFIRMED` to `yes` only for reviewed T1 scans.
3. Convert one reviewed DICOM series at a time with `scripts\convert_dicom_series.py`, or use an already available NIfTI scan.
4. Run `py -3.14 .\scripts\train_multimodal.py` to train the exploratory 3D CNN + clinical MLP prototype.

The training script requires at least 20 eligible participants as a technical guard, but that is not an adequate sample-size guarantee. It canonicalizes orientation, normalizes intensity, and resizes volumes; skull stripping, registration, ROI extraction, cross-validation, Grad-CAM, and SHAP are not implemented yet. Do not treat prototype metrics as validated performance.

Keep ADNI participant-level data and MRI files on approved local storage. Review the cohort labels and MRI selections before modeling; this project is for research, not clinical diagnosis.
