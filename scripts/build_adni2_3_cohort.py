"""Build a phase-restricted ADNI2/ADNI3 36-month MCI conversion cohort.

This is separate from build_cohort_from_tables.py and writes a separate output.
It selects a true baseline (VISCODE/VISCODE2=bl) in ADNI2 or ADNI3, then follows
the participant across later phase records by PTID and actual exam date.
Screening (sc) and rollover/initial visits are not treated as new baselines.

Run from the project folder:
  py -3.14 .\scripts\build_adni2_3_cohort.py

Validate DXSUM coding against the downloaded phase-specific data dictionary
before using the output for a final analysis. This script prints aggregate
counts only; participant-level output remains local.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from build_cohort_from_tables import (
    read_table, flag_is_yes, parse_dates, closest_baseline_value,
)

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "metadata"
OUT = META / "cohort_adni2_3_36m.csv"
FOLLOWUP_MONTHS = 36
INCLUDED_PHASES = {"ADNI2", "ADNI3"}


def phase_key(series: pd.Series) -> pd.Series:
    return series.astype("string").str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)


def main() -> None:
    dx = read_table("DXSUM")
    demo = read_table("PTDEMOG")
    mmse = read_table("MMSE")
    adas = read_table("ADAS")

    required = {"DXMCI", "DXAD", "DXNORM"}
    missing = sorted(required - set(dx.columns))
    if missing:
        raise ValueError(f"DXSUM is missing diagnosis flags: {', '.join(missing)}")
    phase_col = "COLPROT" if "COLPROT" in dx.columns else "ORIGPROT" if "ORIGPROT" in dx.columns else None
    if phase_col is None:
        raise ValueError("DXSUM needs COLPROT (preferred) or ORIGPROT to restrict the baseline phase.")
    visit_col = "VISCODE2" if "VISCODE2" in dx.columns else "VISCODE" if "VISCODE" in dx.columns else None
    if visit_col is None:
        raise ValueError("DXSUM needs VISCODE2 or VISCODE to identify baseline visits.")

    dx["_date"] = parse_dates(dx, "EXAMDATE")
    dx = dx.dropna(subset=["PTID", "_date"]).copy()
    dx["_phase"] = phase_key(dx[phase_col])
    dx["_visit"] = dx[visit_col].astype("string").str.strip().str.lower()
    for col in ["DXMCI", "DXAD", "DXNORM"]:
        dx[f"_{col}"] = flag_is_yes(dx[col])

    print("Phase values in DXSUM (aggregate counts only):")
    print(dx[phase_col].astype("string").fillna("<missing>").value_counts().head(30).to_string())
    baseline_candidates = dx[
        dx["_phase"].isin(INCLUDED_PHASES)
        & dx["_visit"].eq("bl")
        & dx["_DXMCI"]
    ].copy()
    if baseline_candidates.empty:
        raise ValueError(
            "No baseline MCI rows found for ADNI2/ADNI3 using COLPROT/ORIGPROT and VISCODE=bl. "
            "Check the local DXSUM phase and visit coding against DATADIC; no participant rows were printed."
        )

    # A participant contributes one index baseline only, even if they roll over
    # across phases. The earliest eligible ADNI2/ADNI3 baseline is retained.
    baseline = (
        baseline_candidates.sort_values("_date").drop_duplicates("PTID", keep="first")
        .rename(columns={"_date": "baseline_date", "_phase": "baseline_phase", "_visit": "_baseline_visit"})
        [["PTID", "baseline_date", "baseline_phase", "_baseline_visit"]]
        .copy()
    )

    outcomes = []
    for row in baseline.itertuples(index=False):
        pid, t0, phase, visit = row
        horizon = t0 + pd.DateOffset(months=FOLLOWUP_MONTHS)
        subject_visits = dx[dx["PTID"].eq(pid) & (dx["_date"] >= t0)].sort_values("_date")
        within = subject_visits[subject_visits["_date"] <= horizon]
        ad_rows = within[within["_DXAD"]]
        norm_rows = within[within["_DXNORM"]]
        mci_at_or_after = subject_visits[subject_visits["_DXMCI"] & (subject_visits["_date"] >= horizon)]
        conflicts = within[(within[["_DXMCI", "_DXAD", "_DXNORM"]].sum(axis=1) > 1)]

        if not conflicts.empty:
            label, status = np.nan, "excluded: conflicting diagnosis flags"
        elif not ad_rows.empty:
            first_ad_date = ad_rows["_date"].min()
            if not norm_rows[norm_rows["_date"] <= first_ad_date].empty:
                label, status = np.nan, "excluded: normal diagnosis before/by AD conversion"
            else:
                label, status = 1, "pMCI: AD by 36 months"
        elif not norm_rows.empty:
            label, status = np.nan, "excluded: normal diagnosis within 36 months"
        elif not mci_at_or_after.empty:
            label, status = 0, "sMCI: no AD by 36 months; MCI documented at/after month 36"
        else:
            label, status = np.nan, "excluded: insufficient follow-up to establish 36-month status"
        outcomes.append({
            "PTID": pid, "label": label,
            "outcome": {0: "sMCI", 1: "pMCI"}.get(label, "excluded"),
            "label_status": status,
            "followup_months": round((subject_visits["_date"].max() - t0).days / 30.4375, 1),
        })

    cohort = baseline.merge(pd.DataFrame(outcomes), on="PTID", validate="one_to_one")
    cohort["label"] = pd.to_numeric(cohort["label"], errors="coerce")
    cohort = cohort[cohort["label"].notna()].copy()
    cohort["label"] = cohort["label"].astype(int)

    demo_values = [col for col in ["PTGENDER", "PTEDUCAT", "PTDOBYY"] if col in demo.columns]
    cohort = closest_baseline_value(cohort, demo, "VISDATE", demo_values, "DEMO")
    cohort = closest_baseline_value(cohort, mmse, "VISDATE", ["MMSCORE"], "MMSE")
    adas_fields = [col for col in ["TOTAL13", "TOTSCORE"] if col in adas.columns]
    if not adas_fields:
        raise ValueError("ADAS needs TOTAL13 or TOTSCORE. Check its local export columns.")
    adas_field = "TOTAL13" if "TOTAL13" in adas_fields else "TOTSCORE"
    cohort = closest_baseline_value(cohort, adas, "VISDATE", [adas_field], "ADAS")

    gender = cohort.get("demo_ptgender", pd.Series(index=cohort.index, dtype="object")).astype("string").str.strip().str.casefold()
    cohort["sex"] = gender.map({"1": "Male", "1.0": "Male", "male": "Male",
                                "2": "Female", "2.0": "Female", "female": "Female"})
    education_source = cohort["demo_pteducat"] if "demo_pteducat" in cohort else pd.Series(np.nan, index=cohort.index)
    education = pd.to_numeric(education_source, errors="coerce")
    cohort["education_years"] = education.replace([-1, -4], np.nan)
    birth_source = cohort["demo_ptdobyy"] if "demo_ptdobyy" in cohort else pd.Series(np.nan, index=cohort.index)
    birth_year = pd.to_numeric(birth_source, errors="coerce")
    cohort["age_years_approx"] = cohort["baseline_date"].dt.year - birth_year
    cohort.loc[~cohort["age_years_approx"].between(40, 110), "age_years_approx"] = np.nan
    mmse_source = cohort["mmse_mmscore"] if "mmse_mmscore" in cohort else pd.Series(np.nan, index=cohort.index)
    adas_column = f"adas_{adas_field.lower()}"
    adas_source = cohort[adas_column] if adas_column in cohort else pd.Series(np.nan, index=cohort.index)
    cohort["mmse_score"] = pd.to_numeric(mmse_source, errors="coerce").replace([-1, -4], np.nan)
    cohort["adas_score"] = pd.to_numeric(adas_source, errors="coerce").replace([-1, -4], np.nan)

    META.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(OUT, index=False)
    print("\nADNI2/ADNI3 36-month cohort (aggregate counts only):")
    print("Baseline phase:")
    print(cohort["baseline_phase"].value_counts().to_string())
    print("\nSex by outcome:")
    print(cohort.groupby(["sex", "outcome"], dropna=False).size().rename("participants").to_string())
    print(f"\nIncluded labeled participants: {len(cohort)}")
    print("Missing baseline MMSE:", int(cohort["mmse_score"].isna().sum()))
    print("Missing baseline ADAS:", int(cohort["adas_score"].isna().sum()))
    print("Missing sex:", int(cohort["sex"].isna().sum()))
    print("\nBaseline MCI exclusions:")
    print(pd.Series([item["label_status"] for item in outcomes]).value_counts().to_string())
    print(f"\nSaved local cohort to: {OUT}")
    print("Validate phase-specific diagnosis coding and baseline visit selection before modeling.")


if __name__ == "__main__":
    main()
