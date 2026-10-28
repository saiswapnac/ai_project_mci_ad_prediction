"""Match local ADNI MRI metadata to the clinical cohort.

Run from the project folder:
  py -3.14 .\scripts\match_mri_metadata.py

The script first uses an exact PTID/Subject match. If data/raw/ROSTER.csv is
present, it can also map a numeric RID or a Subject_ID ending in an ADNI-style
_S_<RID> token through the official RID-to-PTID roster crosswalk. It prints
aggregate counts only; the participant-level result remains local.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
META = ROOT / "data" / "metadata"
COHORT_PATH = META / "master_cohort_clinical_only.csv"
MRI_PATH = RAW / "MRI_metadata.csv"
ROSTER_PATH = RAW / "ROSTER.csv"
OUT_PATH = META / "mri_scan_candidates_local.csv"
WINDOW_DAYS = 180


def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"[^A-Z0-9]+", "_", str(c).strip().upper()).strip("_") for c in df.columns]
    return df


def normalize_ptid(value) -> str | None:
    if pd.isna(value):
        return None
    key = re.sub(r"\s+", "", str(value)).strip().upper()
    key = re.sub(r"^ADNI[_-]", "", key)
    return key or None


def read_roster() -> tuple[dict[int, str], set[int]]:
    """Return only unique RID->PTID links and the RIDs ambiguous in the roster."""
    if not ROSTER_PATH.exists():
        return {}, set()
    roster = normalize_headers(pd.read_csv(ROSTER_PATH, low_memory=False))
    if not {"RID", "PTID"}.issubset(roster.columns):
        print("ROSTER.csv is present but does not contain RID and PTID; skipping RID crosswalk.")
        return {}, set()
    roster["_rid"] = pd.to_numeric(roster["RID"], errors="coerce")
    roster["_ptid"] = roster["PTID"].map(normalize_ptid)
    roster = roster.dropna(subset=["_rid", "_ptid"]).copy()
    roster["_rid"] = roster["_rid"].astype(int)
    counts = roster.groupby("_rid")["_ptid"].nunique()
    ambiguous = set(counts[counts > 1].index.astype(int))
    unique = roster[~roster["_rid"].isin(ambiguous)].drop_duplicates("_rid")
    return dict(zip(unique["_rid"], unique["_ptid"])), ambiguous


def rid_from_subject(value, explicit_rid: bool = False) -> int | None:
    """Extract a RID only from a numeric RID or an explicit ADNI _S_<digits> ID."""
    if pd.isna(value):
        return None
    token = re.sub(r"\s+", "", str(value)).strip().upper()
    if explicit_rid and re.fullmatch(r"\d+", token):
        return int(token)
    if re.fullmatch(r"\d+", token):
        return int(token)
    match = re.search(r"_S_(\d{1,5})$", token)
    return int(match.group(1)) if match else None


def local_path(value: Path) -> Path:
    return value if value.is_absolute() else ROOT / value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=COHORT_PATH, help="Local cohort CSV")
    parser.add_argument("--output", type=Path, default=OUT_PATH, help="Local candidate CSV output")
    args = parser.parse_args()
    cohort_path = local_path(args.cohort)
    output_path = local_path(args.output)

    if not cohort_path.exists():
        print(f"Clinical cohort not found: {cohort_path}")
        print("Run scripts/build_cohort_from_tables.py first.")
        return 1
    if not MRI_PATH.exists():
        print(f"MRI metadata CSV not found: {MRI_PATH}")
        print("Save the IDA image-search metadata export as data/raw/MRI_metadata.csv and rerun.")
        return 1

    cohort = normalize_headers(pd.read_csv(cohort_path, low_memory=False))
    mri = normalize_headers(pd.read_csv(MRI_PATH, low_memory=False))
    if not {"PTID", "BASELINE_DATE"}.issubset(cohort.columns):
        print("Clinical cohort needs PTID and BASELINE_DATE columns.")
        return 2

    # IDA exports may label the field Subject, Subject_ID, PTID, or RID.
    subject_col = next((c for c in ["SUBJECT", "SUBJECT_ID", "PTID", "RID"] if c in mri.columns), None)
    required_mri = {"IMAGE_DATA_ID", "VISIT", "MODALITY", "DESCRIPTION", "TYPE", "ACQ_DATE", "FORMAT"}
    missing = sorted(required_mri - set(mri.columns))
    if subject_col is None:
        missing.append("SUBJECT (or SUBJECT_ID/PTID/RID)")
    if missing:
        print("MRI metadata is missing required columns:", ", ".join(missing))
        print("Check the IDA metadata CSV headers locally; this script will not print rows or IDs.")
        return 2

    cohort["_subject_key"] = cohort["PTID"].map(normalize_ptid)
    cohort["BASELINE_DATE"] = pd.to_datetime(cohort["BASELINE_DATE"], errors="coerce")
    cohort = cohort.dropna(subset=["_subject_key", "BASELINE_DATE"]).drop_duplicates("_subject_key")
    cohort_keys = set(cohort["_subject_key"])

    mri["_raw_subject"] = mri[subject_col].astype("string").str.strip()
    mri["_subject_key"] = mri["_raw_subject"].map(normalize_ptid)
    mri["_matched_ptid"] = mri["_subject_key"].where(mri["_subject_key"].isin(cohort_keys))
    mri["_join_method"] = mri["_matched_ptid"].notna().map({True: "exact PTID", False: "unmatched"})
    exact_overlap = set(mri.loc[mri["_matched_ptid"].notna(), "_matched_ptid"])

    rid_to_ptid, ambiguous_roster_rids = read_roster()
    if rid_to_ptid:
        needs_roster = mri["_matched_ptid"].isna()
        rid_values = mri.loc[needs_roster, "_raw_subject"].map(
            lambda value: rid_from_subject(value, explicit_rid=(subject_col == "RID"))
        )
        mapped = rid_values.map(rid_to_ptid)
        valid = mapped.isin(cohort_keys)
        row_indices = mapped.index[valid]
        mri.loc[row_indices, "_matched_ptid"] = mapped.loc[row_indices]
        mri.loc[row_indices, "_join_method"] = "ROSTER RID crosswalk"

    mri["ACQ_DATE"] = pd.to_datetime(mri["ACQ_DATE"], errors="coerce")
    mri["_modality"] = mri["MODALITY"].astype("string").str.strip().str.upper()
    mri["_description"] = mri["DESCRIPTION"].fillna("").astype(str)
    linked_rows = mri[mri["_matched_ptid"].notna()].copy()
    linked = linked_rows.merge(
        cohort[["_subject_key", "PTID", "BASELINE_DATE"]],
        left_on="_matched_ptid", right_on="_subject_key", how="inner", validate="many_to_one"
    )
    linked["days_from_baseline"] = (linked["ACQ_DATE"] - linked["BASELINE_DATE"]).dt.days
    linked["abs_days_from_baseline"] = linked["days_from_baseline"].abs()
    linked["modality_is_mr"] = linked["_modality"].isin({"MR", "MRI"})
    # Text is only a review hint and does not certify T1 weighting.
    linked["description_t1_hint"] = linked["_description"].str.contains(
        r"MPRAGE|IR\s*[-/]?\s*F?SPGR|T1W|T1.WEIGHT", case=False, regex=True, na=False
    )
    candidates = linked[
        linked["modality_is_mr"] & linked["ACQ_DATE"].notna()
        & linked["abs_days_from_baseline"].le(WINDOW_DAYS)
    ].copy()
    candidates = candidates.sort_values(["PTID", "abs_days_from_baseline", "IMAGE_DATA_ID"])

    keep = [c for c in ["PTID", "BASELINE_DATE", "IMAGE_DATA_ID", subject_col, "VISIT", "MODALITY",
                         "DESCRIPTION", "TYPE", "ACQ_DATE", "FORMAT", "days_from_baseline",
                         "abs_days_from_baseline", "description_t1_hint", "_join_method"] if c in candidates.columns]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidates[keep].to_csv(output_path, index=False)

    total_overlap = set(linked["PTID"].map(normalize_ptid))
    roster_overlap = set(linked.loc[linked["_join_method"].eq("ROSTER RID crosswalk"), "PTID"].map(normalize_ptid))
    matched_in_window = set(candidates["PTID"].map(normalize_ptid))
    print("MRI metadata match summary (aggregate counts only):")
    print("Clinical cohort participants:", cohort["_subject_key"].nunique())
    print("MRI metadata rows:", len(mri))
    print("MRI metadata participant labels:", mri["_raw_subject"].nunique(dropna=True))
    print("Exact PTID overlap:", len(exact_overlap))
    print("Additional participants mapped through ROSTER:", len(roster_overlap - exact_overlap))
    print("Total participant-ID overlap after mapping:", len(total_overlap))
    print(f"Participants with MR metadata within +/-{WINDOW_DAYS} days of baseline:", len(matched_in_window))
    print("Candidate scan rows in window:", len(candidates))
    print("Candidate rows with T1-like description text (review required):", int(candidates["description_t1_hint"].sum()))
    if not ROSTER_PATH.exists():
        print("ROSTER.csv not found; RID-to-PTID fallback was unavailable.")
    elif ambiguous_roster_rids:
        print("Ambiguous RID entries in ROSTER were skipped:", len(ambiguous_roster_rids))
    if not candidates.empty:
        print("Format counts:")
        print(candidates["FORMAT"].astype("string").fillna("<missing>").value_counts().to_string())
        print("Visit counts (top 15):")
        print(candidates["VISIT"].astype("string").fillna("<missing>").value_counts().head(15).to_string())
    print(f"\nSaved participant-level scan candidates locally to: {output_path}")
    print("Review visit/date and T1 weighting in IDA. This script does not inspect or certify image pixels.")
    if not total_overlap:
        print("No participant matches. Check that the metadata export covers the cohort and that the correct PTID/RID mapping is available.")
    elif len(total_overlap) < cohort["_subject_key"].nunique():
        print("Some cohort participants have no matching rows in this metadata export; ID matching cannot recover scans absent from the export.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
