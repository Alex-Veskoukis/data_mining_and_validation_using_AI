"""
Implements audit Step 3 from "Governance by Evidence: Regulated Predictors in
Decision-Tree Models" (Section 3.2): confirm each extracted item is a
decision-tree predictor (not an outcome or noise) on the audited DOIs that
passed AI relevance and domain gates. Using stratified Crossref/OpenAlex
weights, this code reports weighted confusion, precision/recall, and Cohen's
kappa that are cited in the paper and feed the A3 factor in the propagated
multiplier S (Section 3.3).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.io import PROC  # keep if you use it for project paths

PROCESSED_DIR = Path(PROC)
INPUT_PATH = PROCESSED_DIR / "validated_features_audited.json"

STRATA_COLS = ["source", "domain"]
SAMPLE_FLAG_COL = "desicion_tree_relevant_audited"

def load_json_df(path: Path) -> pd.DataFrame:
    try:
        return pd.read_json(path, lines=True)
    except ValueError:
        return pd.read_json(path)

def require_cols(df: pd.DataFrame, cols):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}")

def norm_bool_from_labels(series: pd.Series, true_label: str, false_label: str) -> pd.Series:
    s = series.astype("string").str.strip().str.lower()
    true_norm  = true_label.strip().lower()
    false_norm = false_label.strip().lower()
    out = pd.Series(np.nan, index=series.index, dtype="float")
    out[s == true_norm]  = True
    out[s == false_norm] = False
    return out

def compute_audit_weights(df: pd.DataFrame, sample_mask: pd.Series, strata_cols: list) -> pd.Series:
    w = pd.Series(np.nan, index=df.index, dtype=float)
    Npop = len(df)
    Nsam = int(sample_mask.sum())
    if Nsam == 0:
        raise ValueError("Sample size is zero; check 'desicion_tree_relevant_audited' values.")
    use_cols = [c for c in strata_cols if c in df.columns]
    if use_cols:
        pop = df.groupby(use_cols, dropna=False, observed=False).size().reset_index(name="Npop")
        sam = df.loc[sample_mask].groupby(use_cols, dropna=False, observed=False).size().reset_index(name="Nsam")
        tbl = sam.merge(pop, on=use_cols, how="left")
        tbl["weight"] = tbl["Npop"] / tbl["Nsam"]
        sampled = df.loc[sample_mask, use_cols].reset_index()
        sampled = sampled.merge(tbl[use_cols + ["weight"]], on=use_cols, how="left")
        w.loc[sample_mask] = sampled["weight"].to_numpy()
    else:
        w.loc[sample_mask] = Npop / Nsam
    return w

def kappa_from_weighted_counts(tp, fp, fn, tn):
    """Cohen's kappa for nominal 2x2 from weighted counts."""
    W = tp + fp + fn + tn
    if not np.isfinite(W) or W <= 0:
        return np.nan, np.nan, np.nan, 0.0
    po = (tp + tn) / W
    ai_pos, ai_neg = tp + fp, tn + fn
    hu_pos, hu_neg = tp + fn, tn + fp
    pe = (ai_pos / W) * (hu_pos / W) + (ai_neg / W) * (hu_neg / W)
    denom = 1.0 - pe
    if denom <= 0:
        return np.nan, po, pe, W
    kappa = (po - pe) / denom
    return float(kappa), float(po), float(pe), float(W)

# ----------------------------
# Load
# ----------------------------
df = load_json_df(INPUT_PATH)
require_cols(
    df,
    [
        "doi",
        SAMPLE_FLAG_COL,
        "decision_trees_related",
        "domain",
        "domain_validated",
        "feature_validation",
        "feature_validation_human_verified",
    ],
)

# ----------------------------
# Routing masks and weights (on THIS file only)
# ----------------------------
in_sample = df[SAMPLE_FLAG_COL] == 1
gate_relevance = df["decision_trees_related"].astype("string").str.strip().str.lower().eq("relevant")
gate_domain    = df["domain_validated"].astype("string").str.strip().str.lower() \
                   .eq(df["domain"].astype("string").str.strip().str.lower())

audit_weight = compute_audit_weights(df, in_sample, STRATA_COLS)

# Evaluation set: audited DOIs that pass both AI gates
mask_eval = in_sample & gate_relevance & gate_domain

# Normalize labels to booleans; unknowns -> NaN
ai_valid = norm_bool_from_labels(df.loc[mask_eval, "feature_validation"], "Valid", "Not valid")
hu_valid = norm_bool_from_labels(df.loc[mask_eval, "feature_validation_human_verified"], "Valid", "Not valid")

# Keep only rows with both AI and Human labels
m = ai_valid.notna() & hu_valid.notna()
w = audit_weight.loc[mask_eval][m].astype(float)
ai = ai_valid[m].astype(bool)
hu = hu_valid[m].astype(bool)

if w.empty:
    raise ValueError("No audited rows with both AI and human feature-validation labels after AI gates.")

# ----------------------------
# Weighted confusion and metrics
# ----------------------------
TP3 = float(w[(ai)  & (hu)].sum())     # AI Valid, Human Valid
FP3 = float(w[(ai)  & (~hu)].sum())    # AI Valid, Human Not valid
FN3 = float(w[(~ai) & (hu)].sum())     # AI Not valid, Human Valid
TN3 = float(w[(~ai) & (~hu)].sum())    # AI Not valid, Human Not valid

feature_precision = TP3 / (TP3 + FP3) if (TP3 + FP3) > 0 else np.nan
feat_rec          = TP3 / (TP3 + FN3) if (TP3 + FN3) > 0 else np.nan
n_audited         = float(w.sum())

# ----------------------------
# NEW: Cohen's kappa on the gated evaluation set
# ----------------------------
kappa3, agreement3, chance_agreement3, weight_total3 = kappa_from_weighted_counts(TP3, FP3, FN3, TN3)

# Optional cross-check via scikit-learn if available
try:
    from sklearn.metrics import cohen_kappa_score
    kappa3_sklearn = float(cohen_kappa_score(hu.to_numpy(), ai.to_numpy(), sample_weight=w.to_numpy()))
except Exception:
    kappa3_sklearn = np.nan

# ----------------------------
# Save outputs
# ----------------------------
pd.DataFrame([{
    "TP3": TP3,
    "FP3": FP3,
    "FN3": FN3,
    "TN3": TN3,
    "feature_precision": feature_precision,
    "feat_rec":  feat_rec,
    "n_audited": n_audited,

    # NEW reliability outputs
    "kappa3": kappa3,
    "agreement3": agreement3,
    "chance_agreement3": chance_agreement3,
    "weight_total3": weight_total3,
    "n_valid_pairs3": int(m.sum()),
    "kappa3_sklearn_check": kappa3_sklearn
}]).to_csv("ai_vs_human_step3.csv", index=False)

conf = pd.DataFrame(
    {
        "Human_Not_valid": [TN3, FP3],
        "Human_Valid":     [FN3, TP3],
    },
    index=pd.Index(["AI_Not_valid", "AI_Valid"], name="AI")
)
conf.to_csv("feature_validation_confusion.csv")

print(pd.read_csv("ai_vs_human_step3.csv").to_string(index=False))
print("\nSaved: feature_validation_confusion.csv")

# Sanity check
if np.isfinite(kappa3) and -1.0 <= kappa3 <= 1.0 and 0.0 <= agreement3 <= 1.0 and 0.0 <= chance_agreement3 <= 1.0:
    print("\n[OK] Cohen's kappa computed and within bounds. agreement3 and chance_agreement3 in [0,1].")
else:
    print("\n[WARN] Kappa or agreements out of bounds or NaN. Check inputs and weights.")
