"""
Implements audit Step 5 from "Governance by Evidence: Regulated Predictors in
Decision-Tree Models" (Section 3.2): evaluate regulation status plus confidence
at the feature x regulation pair level on the audited sample (propagated sample
flag). Applying the same AI gates and stratified weights as earlier steps, this
code computes purity of AI Regulated+High against human labels overall and per
regulation id; these results are reported in the paper and supply the A5 factor
for the propagated multiplier S (Section 3.3).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.io import PROC  # keep if you use project paths

PROCESSED_DIR = Path(PROC)
INPUT_PATH = PROCESSED_DIR / "validated_feature_regulation_audited.json"

# Set to the exact stratification used for the 1,000 DOI sample. Use [] if none/unknown.
STRATA_COLS = ["source", "domain"]

# ----------------------------
# Helpers
# ----------------------------
def load_json_df(path: Path) -> pd.DataFrame:
    try:
        return pd.read_json(path, lines=True)
    except ValueError:
        return pd.read_json(path)

def require_cols(df: pd.DataFrame, cols):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}")

def find_sample_flag(df: pd.DataFrame) -> str:
    for c in ["feature_validation_audited", "desicion_tree_relevant_audited"]:
        if c in df.columns:
            return c
    raise ValueError("No sample-flag column found. Expected 'feature_validation_audited' or 'desicion_tree_relevant_audited'.")

def norm_yes(series: pd.Series, true_label: str) -> pd.Series:
    s = series.astype("string").str.strip().str.lower()
    return s.eq(true_label.strip().lower())

def norm_eq(a: pd.Series, b: pd.Series) -> pd.Series:
    return a.astype("string").str.strip().str.lower().eq(b.astype("string").str.strip().str.lower())

def norm_reg_high_ai(df: pd.DataFrame) -> pd.Series:
    s_reg = df["regulation_status"].astype("string").str.strip().str.lower()
    s_conf = df["confidence"].astype("string").str.strip().str.lower()
    return s_reg.eq("regulated") & s_conf.eq("high")

def norm_reg_high_human(df: pd.DataFrame) -> pd.Series:
    s_reg = df["regulation_status_human_verified"].astype("string").str.strip().str.lower()
    s_conf = df["confidence_human_verified"].astype("string").str.strip().str.lower()
    # rows lacking human labels become False here; we'll also mask them out explicitly
    return s_reg.eq("regulated") & s_conf.eq("high")

def compute_audit_weights(df: pd.DataFrame,
                          sample_mask: pd.Series,
                          strata_cols: list) -> pd.Series:
    """
    Weight per sampled row = population_count(stratum) / sample_count(stratum).
    Non-sampled rows get NaN. Robust to missing strata columns.
    """
    w = pd.Series(np.nan, index=df.index, dtype=float)
    Npop = len(df)
    Nsam = int(sample_mask.sum())
    if Nsam == 0:
        raise ValueError("Sample size is zero at pair level; check the sample-flag values.")

    use_cols = [c for c in strata_cols if c in df.columns]
    if use_cols:
        pop = (df.groupby(use_cols, dropna=False, observed=False)
                 .size().reset_index(name="Npop"))
        sam = (df.loc[sample_mask]
                 .groupby(use_cols, dropna=False, observed=False)
                 .size().reset_index(name="Nsam"))
        tbl = sam.merge(pop, on=use_cols, how="left")
        tbl["weight"] = tbl["Npop"] / tbl["Nsam"]

        sampled = df.loc[sample_mask, use_cols].reset_index()
        sampled = sampled.merge(tbl[use_cols + ["weight"]], on=use_cols, how="left")
        w.loc[sample_mask] = sampled["weight"].to_numpy()
    else:
        w.loc[sample_mask] = Npop / Nsam
    return w

# ----------------------------
# Load
# ----------------------------
df = load_json_df(INPUT_PATH)
require_cols(
    df,
    [
        "doi",
        "reg_id",
        "regulation_status",
        "confidence",
        "regulation_status_human_verified",
        "confidence_human_verified",
    ],
)

SAMPLE_FLAG_COL = find_sample_flag(df)

# ----------------------------
# Routing masks (AI-only gates; missing columns => gate passes)
# ----------------------------
in_sample = df[SAMPLE_FLAG_COL] == 1

# Gate 1: AI relevance
gate_rel = True
if "decision_trees_related" in df.columns:
    gate_rel = norm_yes(df["decision_trees_related"], "Relevant")

# Gate 2: AI domain equals searched domain
gate_dom = True
if ("domain_validated" in df.columns) and ("domain" in df.columns):
    gate_dom = norm_eq(df["domain_validated"], df["domain"])

# Gate 3: AI feature is valid
gate_feat = True
T = "feature_validation" in df.columns  # guard for naming clarity; python variable accepted with unicode
if "feature_validation" in df.columns:
    s = df["feature_validation"].astype("string").str.strip().str.lower()
    gate_feat = s.eq("valid")

# Evaluation mask: audited pairs that came from AI-gated path
mask_eval = in_sample & gate_rel & gate_dom & gate_feat

# Human labels must exist to evaluate
has_human = df["regulation_status_human_verified"].notna() & df["confidence_human_verified"].notna()
mask_eval = mask_eval & has_human

if not mask_eval.any():
    raise ValueError("No audited pair rows with human labels after applying AI gates.")

# ----------------------------
# Weights and boolean flags
# ----------------------------
audit_weight = compute_audit_weights(df, in_sample, STRATA_COLS)

ai_reg_high = norm_reg_high_ai(df)
hu_reg_high = norm_reg_high_human(df)

w = audit_weight[mask_eval].astype(float)
ai = ai_reg_high[mask_eval]
hu = hu_reg_high[mask_eval]
regs = df.loc[mask_eval, "reg_id"].astype("string").str.strip()

# ----------------------------
# Overall precision for AI Regulated+High
# ----------------------------
denom = float(w[ai].sum())                       # weighted AI RegHigh mass within evaluated rows
agree = float(w[ai & hu].sum())                  # weighted intersection AI RegHigh and Human RegHigh
feature_regulated_precsion = agree / denom if denom > 0 else np.nan

pd.DataFrame([{
    "denom_AI_RegHigh": denom,
    "agree_RegHigh": agree,
    "feature_regulated_precsion": feature_regulated_precsion
}]).to_csv("ai_vs_human_step5.csv", index=False)

# ----------------------------
# Per-reg_id precision
# ----------------------------
dfr = pd.DataFrame({"w": w, "ai": ai, "hu": hu, "reg": regs})

den_by_reg = dfr.loc[dfr["ai"]].groupby("reg", observed=False, dropna=False)["w"].sum()
agr_by_reg  = dfr.loc[dfr["ai"] & dfr["hu"]].groupby("reg", observed=False, dropna=False)["w"].sum()

out = pd.DataFrame({
    "denom_AI_RegHigh": den_by_reg
}).join(agr_by_reg.rename("agree_RegHigh")).fillna(0.0)
out["final_prec_by_reg"] = np.where(out["denom_AI_RegHigh"] > 0,
                                    out["agree_RegHigh"] / out["denom_AI_RegHigh"],
                                    np.nan)
out.reset_index().rename(columns={"reg": "reg_id"}).to_csv("final_precision_by_reg.csv", index=False)

# Optional console prints
print(pd.read_csv("ai_vs_human_step5.csv").to_string(index=False))
print("\nSaved: final_precision_by_reg.csv")
