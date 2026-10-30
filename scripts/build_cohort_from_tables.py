"""Build an ADNI MCI-to-AD 36-month cohort from local exported RDA tables.

Run from the project folder:
  py scripts/build_cohort_from_tables.py

This script reads participant-level files locally and writes derived participant-level
outputs locally. Do not upload the data or its row-level output to public services.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "metadata" / "master_cohort_clinical_only.csv"
FOLLOWUP_MONTHS = 36


def read_table(name: str) -> pd.DataFrame:
    path = RAW / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Required file missing: {path}")
    table = pd.read_csv(path, low_memory=False)
    table.columns = table.columns.str.strip().str.upper()
    if "PTID" not in table.columns:
        raise ValueError(f"{path.name} has no PTID column")
    table["PTID"] = table["PTID"].astype("string").str.strip()
    return table


def flag_is_yes(series: pd.Series) -> pd.Series:
    # ADNI diagnostic-summary checkbox fields are coded 1=Yes per the provided dictionary.
    numeric_yes = pd.to_numeric(series, errors="coerce").eq(1)
    text_yes = series.astype("string").str.strip().str.upper().isin({"1", "1.0", "YES", "Y", "TRUE", "T"})
    return numeric_yes | text_yes


def print_safe_dx_diagnostics(dx: pd.DataFrame) -> None:
    """Print category counts only; never prints participant identifiers or rows."""
    print("\nSafe DXSUM diagnostics (category counts only; no participant IDs):")
    for col in ["VISCODE2", "VISCODE", "DXMCI", "DXAD", "DXNORM"]:
        if col in dx.columns:
            counts = dx[col].astype("string").fillna("<missing>").value_counts(dropna=False).head(30)
            print(f"\n{col} value counts:")
            print(counts.to_string())
    visit_col = "VISCODE2" if "VISCODE2" in dx.columns else "VISCODE"
    if visit_col in dx.columns and "DXMCI" in dx.columns:
        visits = dx[visit_col].astype("string").str.strip().str.lower()
        mci_yes = flag_is_yes(dx["DXMCI"])
        print(f"\nDXMCI=Yes counts by {visit_col} (top 30):")
        print(visits[mci_yes].fillna("<missing>").value_counts().head(30).to_string())


def parse_dates(table: pd.DataFrame, column: str) -> pd.Series:
    if column not in table.columns:
        raise ValueError(f"Expected date column {column!r} not found. Columns: {table.columns.tolist()}")
    return pd.to_datetime(table[column], errors="coerce")


def closest_baseline_value(base: pd.DataFrame, table: pd.DataFrame, date_col: str,
                           value_cols: list[str], prefix: str, max_days: int = 180) -> pd.DataFrame:
    """Attach the closest same-visit measure within max_days of each diagnosis baseline."""
    if date_col not in table.columns:
        raise ValueError(f"{prefix}: expected {date_col!r}; available: {table.columns.tolist()}")
    t = table.copy()
    t["_date"] = pd.to_datetime(t[date_col], errors="coerce")
    t = t.dropna(subset=["PTID", "_date"])
    # Require at least one non-null measurement in value_cols if value_cols are present
    valid_cols = [c for c in value_cols if c in t.columns]
    if valid_cols:
        has_val = t[valid_cols].notna().any(axis=1)
        t = t[has_val].copy()
    if "VISCODE2" in t.columns:
        t["_visit_key"] = t["VISCODE2"].astype("string").str.strip().str.lower()
    elif "VISCODE" in t.columns:
        t["_visit_key"] = t["VISCODE"].astype("string").str.strip().str.lower()
    else:
        t["_visit_key"] = ""
    selections = []
    for row in base[["PTID", "baseline_date", "_baseline_visit"]].itertuples(index=False):
        pid, target_date, visit_key = row
        options = t[t["PTID"].eq(pid)].copy()
        options["_delta_days"] = (options["_date"] - target_date).dt.days.abs()
        if visit_key:
            # Prefer an exact visit-code match only when it is also within the date window.
            same_visit = options[options["_visit_key"].eq(visit_key) & (options["_delta_days"] <= max_days)]
            if not same_visit.empty:
                options = same_visit
        options = options[options["_delta_days"] <= max_days].sort_values("_delta_days")
        if options.empty:
            # For fixed demographics, fallback to any available visit for that participant if within a wider window
            fallback = t[t["PTID"].eq(pid)].copy()
            if prefix.upper() == "DEMO" and not fallback.empty:
                fallback["_delta_days"] = (fallback["_date"] - target_date).dt.days.abs()
                options = fallback.sort_values("_delta_days")
        if options.empty:
            selections.append({"PTID": pid})
            continue
        chosen = options.iloc[0]
        item = {"PTID": pid, f"{prefix.lower()}_date": chosen["_date"],
                f"{prefix.lower()}_days_from_baseline": int(chosen["_delta_days"])}
        for col in value_cols:
            item[f"{prefix.lower()}_{col.lower()}"] = chosen[col] if col in chosen.index else np.nan
        selections.append(item)
    selected = pd.DataFrame(selections).drop_duplicates("PTID", keep="first")
    return base.merge(selected, on="PTID", how="left", validate="one_to_one")


def main() -> None:
    dx = read_table("DXSUM")
    demo = read_table("PTDEMOG")
    mmse = read_table("MMSE")
    adas = read_table("ADAS")

    required_dx = {"DXMCI", "DXAD", "DXNORM"}
    missing_dx = required_dx - set(dx.columns)
    if missing_dx:
        raise ValueError(f"DXSUM is missing diagnosis flags {sorted(missing_dx)}")
    dx["_date"] = parse_dates(dx, "EXAMDATE")
    dx = dx.dropna(subset=["PTID", "_date"]).copy()
    for col in ["DXMCI", "DXAD", "DXNORM"]:
        dx[f"_{col}"] = flag_is_yes(dx[col])
    if "VISCODE2" in dx.columns:
        dx["_visit"] = dx["VISCODE2"].astype("string").str.strip().str.lower()
    elif "VISCODE" in dx.columns:
        dx["_visit"] = dx["VISCODE"].astype("string").str.strip().str.lower()
    else:
        dx["_visit"] = ""

    # Select the earliest explicitly coded baseline visit with MCI.
    candidates = dx[dx["_visit"].isin({"bl", "sc"}) & dx["_DXMCI"]].copy()
    if candidates.empty:
        print_safe_dx_diagnostics(dx)
        raise ValueError("No baseline MCI rows recognized under the current rule (VISCODE2/VISCODE bl or sc with DXMCI=1). Use the aggregate diagnostics above to adjust the rule; do not share participant rows or IDs.")
    baseline = (candidates.sort_values("_date").drop_duplicates("PTID", keep="first")
                .rename(columns={"_date": "baseline_date", "_visit": "_baseline_visit"})
                [["PTID", "baseline_date", "_baseline_visit"]].copy())

    # Establish outcomes using diagnosis visits only; baseline features are attached separately.
    outcomes = []
    for row in baseline.itertuples(index=False):
        pid, t0, _visit = row
        end = t0 + pd.DateOffset(months=FOLLOWUP_MONTHS)
        visits = dx[(dx["PTID"].eq(pid)) & (dx["_date"] >= t0)].sort_values("_date").copy()
        within = visits[visits["_date"] <= end]
        ad_visits = within[within["_DXAD"]]
        norm_visits = within[within["_DXNORM"]]
        mci_at_or_after = visits[visits["_DXMCI"] & (visits["_date"] >= end)]
        simultaneous = within[(within[["_DXMCI", "_DXAD", "_DXNORM"]].sum(axis=1) > 1)]

        if not simultaneous.empty:
            label, status = np.nan, "excluded: conflicting diagnosis flags at a visit"
        elif not ad_visits.empty:
            first_ad_date = ad_visits["_date"].min()
            prior_norm = norm_visits[norm_visits["_date"] <= first_ad_date]
            if not prior_norm.empty:
                label, status = np.nan, "excluded: normal diagnosis before/by AD conversion"
            else:
                label, status = 1, "pMCI: AD diagnosis by 36 months"
        elif not norm_visits.empty:
            label, status = np.nan, "excluded: normal diagnosis during 36-month window"
        elif not mci_at_or_after.empty:
            label, status = 0, "sMCI: MCI documented at/after 36 months; no AD by 36 months"
        else:
            label, status = np.nan, "excluded: insufficient follow-up to establish stable MCI"
        last_date = visits["_date"].max()
        outcomes.append({"PTID": pid, "label": label,
                         "outcome": {0: "sMCI", 1: "pMCI"}.get(label, "excluded"),
                         "label_status": status,
                         "followup_months": round((last_date - t0).days / 30.4375, 1)})
    cohort = baseline.merge(pd.DataFrame(outcomes), on="PTID", validate="one_to_one")
    cohort["label"] = pd.to_numeric(cohort["label"], errors="coerce")
    cohort = cohort[cohort["label"].notna()].copy()
    cohort["label"] = cohort["label"].astype(int)

    # Attach baseline demographic, MMSE, and ADAS measurements by participant and nearest matching visit/date.
    demo_values = [c for c in ["PTGENDER", "PTEDUCAT", "PTDOBYY"] if c in demo.columns]
    cohort = closest_baseline_value(cohort, demo, "VISDATE", demo_values, "DEMO")
    cohort = closest_baseline_value(cohort, mmse, "VISDATE", ["MMSCORE"], "MMSE")
    adas_candidates = [c for c in ["TOTAL13", "TOTSCORE"] if c in adas.columns]
    if not adas_candidates:
        raise ValueError("ADAS needs TOTAL13 or TOTSCORE. Check the ADAS export columns.")
    # Prefer the harmonized 13-item total, which has a consistent clinical interpretation across phases.
    adas_field = "TOTAL13" if "TOTAL13" in adas.columns else "TOTSCORE"
    cohort = closest_baseline_value(cohort, adas, "VISDATE", [adas_field], "ADAS")

    # ADNI gender coding per the supplied dictionary: 1=male, 2=female.
    gender_col = "demo_ptgender"
    if gender_col in cohort.columns:
        sex_value = cohort[gender_col].astype("string").str.strip().str.casefold()
        cohort["sex"] = sex_value.map({
            "1": "Male", "1.0": "Male", "male": "Male",
            "2": "Female", "2.0": "Female", "female": "Female",
        })
    else:
        cohort["sex"] = np.nan
    if "demo_pteducat" in cohort.columns:
        cohort["education_years"] = pd.to_numeric(cohort["demo_pteducat"], errors="coerce").replace([-1, -4], np.nan)
    if "demo_ptdobyy" in cohort.columns:
        dob_str = cohort["demo_ptdobyy"].astype("string").str.strip()
        birth_year = pd.to_datetime(dob_str, errors="coerce").dt.year
        birth_year = birth_year.fillna(pd.to_numeric(dob_str.str.extract(r"(\d{4})")[0], errors="coerce"))
        cohort["age_years_approx"] = cohort["baseline_date"].dt.year - birth_year
        cohort.loc[~cohort["age_years_approx"].between(40, 110), "age_years_approx"] = np.nan
    cohort["mmse_score"] = pd.to_numeric(cohort.get("mmse_mmscore"), errors="coerce").replace([-1, -4], np.nan)
    cohort["adas_score"] = pd.to_numeric(cohort.get(f"adas_{adas_field.lower()}"), errors="coerce").replace([-1, -4], np.nan)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(OUT, index=False)
    counts = (cohort.groupby(["sex", "outcome"], dropna=False).size()
              .rename("participants").reset_index())
    excluded = pd.DataFrame(outcomes)["label_status"].value_counts(dropna=False)
    print("Four-group counts (unknown sex is shown separately if present):")
    print(counts.to_string(index=False))
    print(f"\nIncluded labeled participants: {len(cohort)}")
    print("Missing baseline MMSE:", int(cohort["mmse_score"].isna().sum()))
    print("Missing baseline ADAS:", int(cohort["adas_score"].isna().sum()))
    print("Missing sex:", int(cohort["sex"].isna().sum()))
    print("\nOutcome/exclusion reasons among baseline MCI:")
    print(excluded.to_string())
    print(f"\nSaved cohort CSV locally to: {OUT}")
    print("Review diagnosis coding and visit/date matches before modeling; no MRI data was used.")

if __name__ == "__main__":
    main()


