"""
Implements the population correction from "Governance by Evidence: Regulated
Predictors in Decision-Tree Models" (Section 3.3): apply the conservative
stage-wise multiplier S to AI Regulated+High pair counts. Reads the full
validated_feature_regulation_audited.json population plus the audit summaries
from steps 1-5 (precision of relevance, domain, predictor validity, RDC match,
and status) and multiplies them to produce corrected totals overall and per
regulation. This is the code used for the "Final propagated correction"
equations for T_corr and S reported in the paper.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.io import PROC  # keep if you use project path roots

PROCESSED_DIR = Path(PROC)
PAIRS_PATH = PROCESSED_DIR / "validated_feature_regulation_audited.json"

# ----------------------------
# Helpers
# ----------------------------
def load_json_df(path: Path) -> pd.DataFrame:
    try:
        return pd.read_json(path, lines=True)
    except ValueError:
        return pd.read_json(path)

def norm_reg_high_ai(df: pd.DataFrame) -> pd.Series:
    reg = df["regulation_status"].astype("string").str.strip().str.lower()
    conf = df["confidence"].astype("string").str.strip().str.lower()
    return reg.eq("regulated") & conf.eq("high")

# ----------------------------
# 1) Load full-pop pairs and get AI "Regulated+High" counts
# ----------------------------
pairs = load_json_df(PAIRS_PATH)
for col in ["regulation_status", "confidence", "reg_id"]:
    if col not in pairs.columns:
        raise ValueError(f"Missing required column in pairs file: {col}")

mask_ai_high = norm_reg_high_ai(pairs)

T_AI_high = int(mask_ai_high.sum())
T_AI_high_by_reg = (
    pairs.loc[mask_ai_high, "reg_id"]
         .astype("string").str.strip()
         .value_counts()
         .rename_axis("reg_id").rename("T_AI_high_by_reg")
         .sort_index()
)

# ----------------------------
# 2) Load measured rates (overall + per-reg_id)
# ----------------------------
s1 = pd.read_csv("ai_vs_human_step1.csv")      # has 'prec1'
s2 = pd.read_csv("ai_vs_human_step2.csv")      # has 'pass_ok'
s3 = pd.read_csv("ai_vs_human_step3.csv")      # has 'feat_prec'
s4 = pd.read_csv("ai_vs_human_step4.csv")      # has 'aclass_match'
s5 = pd.read_csv("ai_vs_human_step5.csv")      # has 'final_prec'
byreg = pd.read_csv("final_precision_by_reg.csv")  # 'reg_id','denom_AI_RegHigh','agree_RegHigh','final_prec_by_reg'

prec1       = float(s1.loc[0, "decision_tree_relevance_precision"])
pass_ok     = float(s2.loc[0, "domain_precision"])
feat_prec   = float(s3.loc[0, "feature_precision"])
aclass_match= float(s4.loc[0, "attribute_class_precision"])
final_prec  = float(s5.loc[0, "feature_regulated_precsion"])

# Per-reg_id precision map with fallback to overall final_prec
byreg_map = {str(r): float(p) for r, p in zip(byreg["reg_id"].astype(str), byreg["final_prec_by_reg"])}

# ----------------------------
# 3) Compute conservative corrected totals
# ----------------------------
m_other = prec1 * pass_ok * feat_prec * aclass_match  # upstream precision factors
overall_multiplier = m_other * final_prec

T_conservative = T_AI_high * overall_multiplier

df_overall = pd.DataFrame([{
    "T_AI_high": T_AI_high,
    "decision_tree_relevance_precision": prec1,
    "domain_precision": pass_ok,
    "feature_precision": feat_prec,
    "attribute_class_precision": aclass_match,
    "feature_regulated_precsion": final_prec,
    "multiplier_other": m_other,                 # <- add
    "overall_multiplier": overall_multiplier,
    "T_conservative": T_conservative
}])

df_overall.to_csv("final_counts_corrected.csv", index=False)

# Per-reg_id
rows = []
for reg, cnt in T_AI_high_by_reg.items():
    prec_reg = byreg_map.get(str(reg), final_prec)  # fallback to overall if missing
    mult_reg = m_other * prec_reg
    rows.append({
        "reg_id": reg,
        "T_AI_high_by_reg": int(cnt),
        "final_prec_by_reg": prec_reg,
        "multiplier_other": m_other,
        "overall_multiplier_reg": mult_reg,
        "T_conservative_by_reg": cnt * mult_reg
    })
pd.DataFrame(rows).sort_values("reg_id").to_csv("final_counts_corrected_by_reg.csv", index=False)

# Optional: console print
print(df_overall.to_string(index=False))
print("\nSaved: final_counts_corrected_by_reg.csv")
