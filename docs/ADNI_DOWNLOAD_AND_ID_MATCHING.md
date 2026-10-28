# ADNI download and participant-matching guide

This guide walks through downloading the clinical tables and baseline structural MRI needed by this project, then matching them locally. IDA screens can change, and the exact controls visible may depend on account permissions. Use the official [IDA User Manual](https://adni.loni.usc.edu/support/?core=Data+Sharing+and+Publications+Committee) if your screen differs.

## Read this before downloading

1. Use only an ADNI account with approved data access. The IDA is the authorized location for ADNI downloads. See the [ADNI access and download guide](https://adni.loni.usc.edu/quick-start-guide-asset/getting_started.html).
2. Keep the data on your computer or institution-approved, access-controlled storage. The ADNI Data Use Agreement restricts sharing participant-level data and the use of public-facing AI tools unless data containment is guaranteed. Do not paste participant IDs, table rows, MRI files, derived participant-level tables, or screenshots containing those details into this chat, public repositories, or public AI services. Read the [ADNI Data Use Agreement](https://adni.loni.usc.edu/wp-content/themes/adni_2023/documents/ADNI_Data_Use_Agreement.pdf).
3. Use current IDA releases. ADNI announced in March 2026 that data for 78 participants were removed and should not be used. If you cannot verify when your tables or images were downloaded, get fresh copies from IDA and rebuild the cohort. See the [ADNI Data Advisory](https://adni.loni.usc.edu/news-publications/adni-data-advisory/).
4. Decide which ADNI phases your project includes and use compatible table releases. ADNI phases differ in variables, visit conventions, scanners, and imaging protocols. Record the phase(s), the download date, and the filenames locally. ADNI recommends reporting the data download date because records can be updated.

## Part A — Download the clinical tables

The current project uses separate CSV tables; you do **not** need to find `ADNIMERGE` to continue. The builder expects `DXSUM`, `PTDEMOG`, `MMSE`, and `ADAS`.

### A1. Open the Study Files area

1. Open the [LONI IDA](https://ida.loni.usc.edu/) in a browser and sign in.
2. In the top navigation, choose **Search & Download**.
3. Choose **Study Files**. This is the tabular-data area, separate from the MRI image archive. The official guide describes the left panel as the category navigation and the right panel as the files list. A search box above the categories can search file titles. See [Study Files in the ADNI guide](https://adni.loni.usc.edu/quick-start-guide-asset/study_files.html).
4. If prompted to choose a study, choose **ADNI**. Do not choose only ADNI1 unless your project is intentionally limited to that phase.

### A2. Find and download each required table

Use the left category tree or the Study Files search box. Search the table name if you do not see it in the category list. Select the current release of the table, then use the file's **Download** control. If the interface supports selecting several tables before downloading, you can download them together.

| Download | Likely Study Files category | Purpose in this project |
|---|---|---|
| `DXSUM` / Diagnosis Summary | Assessments → Diagnosis | Baseline diagnosis and longitudinal diagnosis dates used to define the target label |
| `PTDEMOG` / Participant Demographics | Subject Characteristics → Subject Demographics | Sex, education, and birth year/age fields |
| `MMSE` | Assessments → Neuropsychological | Baseline `MMSCORE` |
| `ADAS` | Assessments → Neuropsychological | Baseline `TOTAL13` or the ADAS score selected for the analysis |
| `ROSTER` | Enrollment | Maps `RID` to `PTID` when a table uses RID instead of PTID |
| Data dictionary (`DATADIC`) | Study Info → Data & Database | Variable definitions and coding; use it to verify each field for the downloaded phase |

The locations above follow the organization in ADNI's [file tour](https://adni.loni.usc.edu/quick-start-guide-asset/filetour.html). The category names may differ slightly in a future IDA release, so search by the table name. The dictionary is an important first download because variable definitions can vary by phase; see [ADNI's data dictionary guide](https://adni.loni.usc.edu/quick-start-guide-asset/datadic.html).

### A3. Save the downloaded files locally

1. Extract the downloaded ZIP archive(s) on your computer.
2. Copy the current CSV exports into:

   ```text
   mci_ad_prediction/data/raw/
   ```

3. Ensure the four required files are named exactly:

   ```text
   DXSUM.csv
   PTDEMOG.csv
   MMSE.csv
   ADAS.csv
   ```

4. Save the roster as `ROSTER.csv` and the data dictionary as `DATADIC.csv` if you downloaded them. The current cohort script does not require the roster when all four input tables already contain PTID, but the roster is the official crosswalk if you encounter RID-only records.
5. Keep the original downloads and note the download date in your project notes. Do not combine a table from an old package/archive with a newer table without checking versions and phases.
6. If IDA gave you `ROSTER.rda` rather than `ROSTER.csv`, put it in `data/raw/` and run the converter below. It now supports ROSTER and skips CSVs that already exist unless you explicitly pass `--overwrite`:

   ```powershell
   py -3.14 .\scripts\convert_adni_rda.py .\data\raw --output .\data\raw
   ```

   Confirm `data/raw/ROSTER.csv` exists locally. Do not add `--overwrite` unless you intend to replace existing tables.

## Part B — Build the clinical MCI cohort

Open VS Code's terminal and change to the project folder:

```powershell
cd "C:\Users\saich\Documents\Codex\2026-10-13\can-x20\outputs\mci_ad_prediction"
```

Ask the local project helper to check the files:

```powershell
py -3.14 .\scripts\project_agent.py status
```

Then build the clinical cohort:

```powershell
py -3.14 .\scripts\project_agent.py cohort
```

This runs `build_cohort_from_tables.py` and writes:

```text
data/metadata/master_cohort_clinical_only.csv
```

This is a local participant-level file. Review it only in your approved local environment. Verify diagnosis coding and the exact pMCI/sMCI definition with your supervisor before using labels in a final model. The current code intends pMCI to mean AD diagnosed within 36 months and sMCI to mean no AD diagnosis within 36 months with MCI documented at/after month 36.

## Part C — Search IDA for baseline T1 MRI

### C1. Open the image search area

1. In IDA, choose **Search & Download → Image Collections** (some IDA versions may label this **Image Data** or open an image-search page directly).
2. Choose the **Search** tab for a basic query or **Advanced Search** for more filters. The official image-collection instructions show search fields for subject and image criteria and explain that search results list images by subject. See the [IDA image collection instructions](https://adni.loni.usc.edu/wp-content/uploads/2012/08/instruction-how-to-use-IDA.pdf).
3. Confirm you are searching the **ADNI** study.

### C2. Narrow the search to structural MRI at baseline

1. Set **Modality** to MR/MRI.
2. Set **Visit** to baseline (`bl`) when that filter is available. If the search interface separates phases or study visits, include the phases selected for your clinical cohort.
3. For the subject filter, use the cohort's PTID list if the interface accepts it. If it accepts one ID at a time, use the subject-search field per participant or in supported batches. Another option is to export metadata for all ADNI baseline MR results, then do the PTID/date matching locally; that downloads metadata, not every image.
   - Check the **phase/study** field in your current metadata. The previously used folder names included `adni_1`; if that search was restricted to ADNI1 while the clinical cohort includes several phases, the MRI search itself would omit most of the cohort. Search all phases included by your clinical tables (or deliberately rebuild a phase-restricted clinical cohort). Do not select only ADNI1 unless the cohort is also limited to ADNI1.
4. Search for structural 3D T1 series. Names often include **MPRAGE** or **IR-SPGR**, but names vary by scanner/vendor. Do not accept/reject a scan from its description alone. ADNI says to inspect DICOM header parameters; when multiple series are available, use the QC-selected “chosen” series where available or review the quality rating. See the [ADNI MRI FAQ](https://adni.loni.usc.edu/help-faqs/faqs/).
5. If the site shows a result count, verify that the query is not returning PET, DTI, fMRI, or every visit. Do not download every MRI sequence for every participant.

### C3. Export the MRI metadata CSV

1. In the search results, select the option to export or download the **search results/metadata as CSV**. This CSV is an index of image series; it is not the MRI image itself.
2. Include these fields if the interface offers field selection:

   ```text
   Image Data ID, Subject, Group, Sex, Age, Visit, Modality,
   Description, Type, Acq Date, Format
   ```

3. If the CSV export does not include all fields, download the result metadata first and check the IDA imaging tables under **Study Files → Imaging** for acquisition or QC details.
4. Save the CSV as:

   ```text
   data/raw/MRI_metadata.csv
   ```

5. Keep the CSV local. It contains participant identifiers and scan-level metadata.

### C4. Add only the intended scan series to a collection and download

1. Review candidate results and select the verified baseline T1 series for the cohort participants.
2. Use **Add to collection** (or the equivalent collection/basket control). If results span multiple pages, repeat selection/add-to-collection on each page; the older official instructions explicitly note this step.
3. Open **Data Collections** or **Image Collections** and confirm the selected scan series are present.
4. Choose the offered image format. Download the selected image series only. If you choose DICOM, keep each series in its own folder. If a suitable ADNI-processed NIfTI is offered and you choose to use it, record that choice and apply the same policy to all participants.
5. Store images locally under a structure such as:

   ```text
   data/processed/nifti/
   ```

   or, for original DICOM before conversion:

   ```text
   data/raw/mri_dicom/
   ```

The CSV metadata export and the image download are two separate actions. A CSV containing `Format=DCM` does not contain viewable MRI pixels; it indicates the format of the corresponding image series.

## Part D — Match scan rows to participants

Use these keys and rules; do not join on age, sex, names, or only the final digits of an identifier.

1. **Participant key:** match clinical `PTID` to image metadata `Subject` or `Subject_ID`.
2. **RID crosswalk:** if a clinical or image field contains RID rather than PTID, use the downloaded `ROSTER` table to map RID ↔ PTID. ADNI's FAQ identifies ROSTER as the crosswalk. [ADNI ID FAQ](https://adni.loni.usc.edu/help-faqs/faqs/)
3. **Scan key:** keep `Image Data ID` as the unique scan/series identifier. It is not the participant key.
4. **Visit/date checks:** compare the image `Visit` and `Acq Date` with the clinical baseline visit and date (`EXAMDATE`/`VISDATE`). Visit codes can vary by phase; ADNI recommends using actual elapsed dates for longitudinal timing and using visit codes together with phase information.
5. **One selected image per person:** choose one quality-reviewed baseline T1 scan per participant. If there are multiple candidate series, document why one was selected. If no valid T1 scan exists, exclude that participant from the multimodal analysis rather than substituting a later/follow-up MRI.
6. **Date window:** the current matcher uses a ±180-day window around clinical baseline. Treat that as a provisional setting: agree on it with your supervisor, document it, and manually inspect borderline cases. Do not assume the script has confirmed T1 weighting or scan quality.

If needed, save the roster as `data/raw/ROSTER.csv`. Then run the local match:

```powershell
py -3.14 .\scripts\project_agent.py match
```

It reads the local cohort and `MRI_metadata.csv`, tries exact PTID matching first, then uses `ROSTER.csv` for supported RID formats. It accepts common MRI subject column names such as `Subject`, `Subject_ID`, `PTID`, or `RID`. It writes candidate rows to `data/metadata/mri_scan_candidates_local.csv` and prints counts only. It does not verify the actual DICOM image content or certify a sequence as T1. This uses the existing pandas installation; no separate ID-matching service is needed.

If exact matching is low but the ROSTER fallback increases it, the issue was identifier format. If both overlap counts stay low and the metadata lists only 26 participants, the metadata export/search itself is incomplete for the 240-person cohort. No library can match participants whose scans are missing from the export; broaden the IDA baseline MRI search and export its full metadata first.

For your current files, first verify whether `MRI_metadata.csv` contains ADNI1 only while the clinical cohort includes multiple phases. That phase-scope mismatch is a likely reason the previous search had metadata for 26 participants but only three matches to the cohort. Check the metadata's phase/study and visit fields locally; do not infer phase from a folder name alone.

Create/update the local review sheet:

```powershell
py -3.14 .\scripts\project_agent.py manifest
```

Open `data/metadata/mri_training_manifest.csv` locally. For each row you select:

- Put the path to the corresponding NIfTI file in `NIFTI_PATH`.
- Set `T1_CONFIRMED` to `yes` only after verifying the series in IDA/header/QC.
- Ensure each participant appears once and the scan is baseline, not follow-up.

The manifest generator preserves manual scan confirmation/path fields when the same scan is still selected after a refresh. It may not preserve your choice if the selected image series changes; review the file after every refresh.

## Part E — Convert DICOM to NIfTI and start training

Install `dcm2niix` on your computer separately and make sure `dcm2niix` is available on PATH. Convert one reviewed series folder at a time (do not point it at the entire ADNI directory):

```powershell
py -3.14 .\scripts\convert_dicom_series.py "D:\path\to\one\DICOM_series" --output .\data\processed\nifti
```

Then copy the corresponding NIfTI path into the local manifest. Check local readiness:

```powershell
py -3.14 .\scripts\project_agent.py status
```

When you have enough reviewed scans and have checked labels, dates, one-scan-per-person, and T1 selection, install the Python dependencies if needed and run:

```powershell
py -3.14 -m pip install -r .\requirements-modeling.txt
py -3.14 .\scripts\project_agent.py train
```

The current training script is an exploratory prototype. Its 20-participant guard is only a technical minimum, not an adequate sample-size claim. It does not yet implement skull stripping, registration, ROI extraction, robust cross-validation, or SHAP. Do not report training metrics as validated performance until the preprocessing and evaluation design are complete.

## The local workflow helper commands

From the project folder, the helper runs these local steps:

```powershell
py -3.14 .\scripts\project_agent.py status
py -3.14 .\scripts\project_agent.py demo
py -3.14 .\scripts\project_agent.py cohort
py -3.14 .\scripts\project_agent.py match
py -3.14 .\scripts\project_agent.py manifest
py -3.14 .\scripts\project_agent.py train
```

`demo` uses synthetic data only. The helper is a local workflow runner, not an autonomous AI predictor and not an IDA downloader.

## Troubleshooting

- **I only downloaded a CSV:** the CSV is scan metadata, not the DICOM/NIfTI image. Add selected scans to an IDA collection and download the image series separately.
- **Metadata lists many images but few cohort matches:** confirm the metadata export covers the full search result, `Subject` values correspond to the cohort PTIDs, and the image search included the right phases/visits.
- **The join finds no matches:** inspect column names locally; check PTID formatting and use ROSTER if one table has RID. Do not send the identifiers here.
- **The matcher finds a scan but it may not be T1:** review sequence/header/QC information in IDA. The text description hint is not proof.
- **There are thousands of `.dcm` files:** DICOM files are often slices within a series. File count is not participant count or scan-series count.
- **Only three participants match:** that is not enough for the planned MRI model. Expand the MRI metadata search/download to the eligible clinical cohort before training. The last summary in this project reported only three baseline candidates.
