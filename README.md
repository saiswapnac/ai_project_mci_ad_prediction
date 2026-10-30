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

## Local workflow helper

`scripts/project_agent.py` is a local command-line helper, not a cloud AI agent. It runs the project scripts on your computer and its `status` command reports file readiness and aggregate counts without printing participant IDs.

```powershell
py -3.14 .\scripts\project_agent.py status
py -3.14 .\scripts\project_agent.py demo
py -3.14 .\scripts\project_agent.py cohort-adni23
py -3.14 .\scripts\project_agent.py match-adni23
py -3.14 .\scripts\project_agent.py manifest-adni23
py -3.14 .\scripts\project_agent.py train-adni23
```

For the exact IDA download and ID-joining steps, see [docs/ADNI_DOWNLOAD_AND_ID_MATCHING.md](docs/ADNI_DOWNLOAD_AND_ID_MATCHING.md).

`cohort-adni23` builds a separate, phase-restricted 36-month cohort at `data/metadata/cohort_adni2_3_36m.csv`. It uses true baseline (`bl`) MCI rows in ADNI2/ADNI3, excludes screening as baseline, and follows later diagnosis records by participant/date. Review its phase-specific diagnosis coding before use; it does not replace the original cohort file.

The matching, manifest, and training commands ending in `-adni23` continue that separate cohort through the MRI pipeline and write separate candidate, manifest, model, and report files. The ADNI2/ADNI3 cohort includes baseline MCI broadly: ADNI2 EMCI and LMCI are included, while ADNI3 has MCI without the EMCI/LMCI split. The subgroup distinction is not imposed across phases.

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

1. Convert the downloaded IDA XML metadata into a new local CSV. This leaves the earlier metadata export untouched:

   ```powershell
   py -3.14 .\scripts\convert_ida_xml_metadata.py
   ```

2. Match that metadata to the clinical cohort and create a separate review sheet:

   ```powershell
   py -3.14 .\scripts\match_mri_metadata.py --mri .\data\raw\MRI_metadata_from_xml.csv --output .\data\metadata\mri_scan_candidates_from_xml.csv
   py -3.14 .\scripts\make_mri_manifest.py --candidates .\data\metadata\mri_scan_candidates_from_xml.csv --output .\data\metadata\mri_training_manifest_from_xml.csv
   ```

3. Extract the DICOM ZIP archives into separate folders under `data\processed\dicom` (not over the ZIP files). Match each scan folder to its metadata by participant and acquisition date. Review the series and its DICOM metadata in IDA; text hints do not certify T1 weighting.
4. Convert one reviewed scan series at a time with `scripts\convert_dicom_series.py`, or use a reviewed NIfTI scan. Fill `NIFTI_PATH` with the local NIfTI path in the new manifest and set `T1_CONFIRMED` to `yes` only after verification.
5. Run `py -3.14 .\scripts\train_multimodal.py --manifest .\data\metadata\mri_training_manifest_from_xml.csv` to train the exploratory 3D CNN + clinical MLP prototype.

### Generate report figures in MATLAB

After the training command completes, it writes `reports/test_predictions.csv` (held-out labels, probabilities, and sex; no participant IDs) and `reports/training_history.csv`. In MATLAB, set the current folder to this project and run:

```matlab
run('matlab/generate_report_figures.m')
```

The script saves high-resolution PNGs and a captions text file in `reports/matlab_figures/`. Figures are based on the model's real held-out output. They are still exploratory single-holdout results, not validated clinical performance; interpret sex-specific plots cautiously when subgroup counts are small. The training-loss plot is not a validation-loss curve.

The training script requires at least 20 eligible participants as a technical guard, but that is not an adequate sample-size guarantee. It canonicalizes orientation, normalizes intensity, and resizes volumes; skull stripping, registration, ROI extraction, cross-validation, Grad-CAM, and SHAP are not implemented yet. Do not treat prototype metrics as validated performance.

Keep ADNI participant-level data and MRI files on approved local storage. Review the cohort labels and MRI selections before modeling; this project is for research, not clinical diagnosis.

to get the sythetic data, run
py -3.14 .\scripts\demo_synthetic.py


