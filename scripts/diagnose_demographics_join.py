"""Print aggregate-only diagnostics for matching DXSUM baseline to PTDEMOG.

Run from the project folder:
  py -3.14 .\\scripts\\diagnose_demographics_join.py

This prints counts and value categories only; it never prints participant IDs or rows.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

def read(name):
    df = pd.read_csv(RAW / name, low_memory=False)
    df.columns = df.columns.str.strip().str.upper()
    df["PTID"] = df["PTID"].astype("string").str.strip()
    return df

def yes(s):
    return pd.to_numeric(s, errors="coerce").eq(1) | s.astype("string").str.strip().str.upper().isin({"1", "1.0", "YES", "Y", "TRUE", "T"})

dx = read("DXSUM.csv")
demo = read("PTDEMOG.csv")
dx["_date"] = pd.to_datetime(dx.get("EXAMDATE"), errors="coerce")
demo["_date"] = pd.to_datetime(demo.get("VISDATE"), errors="coerce")
visit_col = "VISCODE2" if "VISCODE2" in dx.columns else "VISCODE"
dx["_visit"] = dx[visit_col].astype("string").str.strip().str.lower()
base = dx[dx["_visit"].isin({"bl", "sc"}) & yes(dx["DXMCI"])].dropna(subset=["PTID", "_date"]).copy()
base["_rank"] = base["_visit"].map({"bl": 0, "sc": 1})
base = base.sort_values(["PTID", "_rank", "_date"]).drop_duplicates("PTID")
demo_visit_col = "VISCODE2" if "VISCODE2" in demo.columns else ("VISCODE" if "VISCODE" in demo.columns else None)
if demo_visit_col:
    demo["_visit"] = demo[demo_visit_col].astype("string").str.strip().str.lower()

print("Aggregate demographics-join diagnostics (no IDs/rows):")
print("Baseline MCI participants:", base["PTID"].nunique())
print("Unique PTDEMOG participants:", demo["PTID"].nunique())
print("Participant-ID overlap:", base["PTID"].isin(set(demo["PTID"].dropna())).sum())
print("Baseline dates parsed:", int(base["_date"].notna().sum()), "/", len(base))
print("PTDEMOG visit dates parsed:", int(demo["_date"].notna().sum()), "/", len(demo))
if "PTGENDER" in demo.columns:
    print("PTGENDER values (counts):")
    print(demo["PTGENDER"].astype("string").fillna("<missing>").value_counts().head(20).to_string())
print("Baseline visit-code counts:")
print(base["_visit"].value_counts().to_string())
if demo_visit_col:
    print("PTDEMOG visit-code counts (top 20):")
    print(demo["_visit"].value_counts().head(20).to_string())

bins = {"no PTDEMOG rows for participant": 0, "no dated PTDEMOG row": 0,
        "same visit and within 120 days": 0, "within 120 days, other visit code": 0,
        "nearest dated row over 120 days": 0}
nearest_days = []
for row in base[["PTID", "_date", "_visit"]].itertuples(index=False, name=None):
    pid, target, visit = row
    options = demo[demo["PTID"].eq(pid)]
    if options.empty:
        bins["no PTDEMOG rows for participant"] += 1
        continue
    options = options.dropna(subset=["_date"]).copy()
    if options.empty:
        bins["no dated PTDEMOG row"] += 1
        continue
    options["_delta"] = (options["_date"] - target).dt.days.abs()
    nearest_days.append(float(options["_delta"].min()))
    within = options[options["_delta"] <= 120]
    same = within[within["_visit"].eq(visit)] if demo_visit_col else within.iloc[0:0]
    if not same.empty:
        bins["same visit and within 120 days"] += 1
    elif not within.empty:
        bins["within 120 days, other visit code"] += 1
    else:
        bins["nearest dated row over 120 days"] += 1
print("Baseline-to-demographic match categories:")
for key, count in bins.items():
    print(f"  {key}: {count}")
if nearest_days:
    print("Nearest-date distance bins (days):")
    print(pd.cut(pd.Series(nearest_days), [-1, 0, 30, 120, 365, float("inf")],
                 labels=["same day", "1-30", "31-120", "121-365", ">365"]).value_counts(sort=False).to_string())
