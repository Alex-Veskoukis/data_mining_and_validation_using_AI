"""
Implements audit Step 1 from "Governance by Evidence: Regulated Predictors in
Decision-Tree Models" (Section 3.2): quantify the precision of the automated
decision-tree relevance screen on the stratified 1,000-DOI audit
(Crossref/OpenAlex weights). This code produces the weighted confusion,
precision of AI "Relevant", miss rate for AI "Not relevant", and Cohen's kappa
that are reported in the paper and feed the A1 factor used in the propagated
multiplier S (Section 3.3).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.io import RAW, PROC  # keep env imports if you use them elsewhere

PROCESSED_DIR = Path(PROC)
INPUT_PATH = PROCESSED_DIR / "merged_classified_decision_trees_related_audited.json"

# Set to your actual stratification; use [] if none or if columns are missing.
STRATA_COLS = ["source","domain"]
SAMPLE_FLAG_COL = "desicion_tree_relevant_audited"

def load_json_df(path: Path) -> pd.DataFrame:
    try:
        return pd.read_json(path, lines=True)
    except ValueError:
        return pd.read_json(path)

def require_cols(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def norm_rel(series: pd.Series) -> pd.Series:
    """Map labels to True/False. Unknowns -> NaN (silently ignored in metrics)."""
    s = series.astype("string").str.strip().str.lower()
    true_set = {"relevant", "true", "1", "yes", "y"}
    false_set = {"not relevant", "not_relevant", "not-relevant", "irrelevant", "false", "0", "no", "n"}
    out = pd.Series(np.nan, index=series.index, dtype="float")
    out[s.isin(true_set)] = True
    out[s.isin(false_set)] = False
    return out

def compute_audit_weights(df: pd.DataFrame,
                          sample_mask: pd.Series,
                          strata_cols: list) -> pd.Series:
    """Weight per sample row = population_count(stratum) / sample_count(stratum).
       Non-sampled rows get NaN. Robust to missing strata columns and unseen strata."""
    w = pd.Series(np.nan, index=df.index, dtype=float)

    Npop = len(df)
    Nsam = int(sample_mask.sum())
    if Nsam == 0:
        raise ValueError("Sample size is zero; check 'desicion_tree_relevant_audited' values.")

    # Use only strata that exist; else fallback to uniform
    strata_cols = [c for c in strata_cols if c in df.columns]
    if len(strata_cols) > 0:
        pop = (
            df.groupby(strata_cols, dropna=False, observed=False)
              .size()
              .reset_index(name="Npop")
        )
        sam = (
            df.loc[sample_mask]
              .groupby(strata_cols, dropna=False, observed=False)
              .size()
              .reset_index(name="Nsam")
        )
        # Keep only strata present in the sample
        tbl = sam.merge(pop, on=strata_cols, how="left")
        tbl["weight"] = tbl["Npop"] / tbl["Nsam"]

        sampled = df.loc[sample_mask, strata_cols].reset_index()
        sampled = sampled.merge(tbl[strata_cols + ["weight"]], on=strata_cols, how="left")
        w.loc[sample_mask] = sampled["weight"].to_numpy()
    else:
        w.loc[sample_mask] = Npop / Nsam

    return w

def kappa_from_weighted_counts(tp, fp, fn, tn):
    """
    Cohen's kappa (nominal) from weighted 2x2 counts.
    Handles degenerate cases robustly.
    """
    W = tp + fp + fn + tn
    if not np.isfinite(W) or W <= 0:
        return np.nan, np.nan, np.nan, 0.0

    po = (tp + tn) / W  # observed agreement
    ai_pos = tp + fp
    ai_neg = tn + fn
    hu_pos = tp + fn
    hu_neg = tn + fp

    pe = (ai_pos / W) * (hu_pos / W) + (ai_neg / W) * (hu_neg / W)

    denom = 1.0 - pe
    if denom <= 0:
        return np.nan, po, pe, W

    kappa = (po - pe) / denom
    return float(kappa), float(po), float(pe), float(W)

# ----------------------------
# Load and validate
# ----------------------------
df = load_json_df(INPUT_PATH)
require_cols(
    df,
    [
        SAMPLE_FLAG_COL,
        "decision_trees_related",
        "decision_trees_related_human_verified",
    ],
)

# ----------------------------
# Build masks and weights
# ----------------------------
in_sample = df[SAMPLE_FLAG_COL] == 1
ai_rel = norm_rel(df["decision_trees_related"])
hu_rel = norm_rel(df["decision_trees_related_human_verified"])

audit_weight = compute_audit_weights(df, in_sample, STRATA_COLS)

# Use only sampled rows with valid AI and Human labels
mask_valid = in_sample & ai_rel.notna() & hu_rel.notna()
w = audit_weight.loc[mask_valid].astype(float)
ai_s = ai_rel.loc[mask_valid].astype(bool)
hu_s = hu_rel.loc[mask_valid].astype(bool)

# ----------------------------
# Weighted confusion counts
# ----------------------------
TP1 = float(w[(ai_s) & (hu_s)].sum())                 # AI Relevant, Human Relevant
FP1 = float(w[(ai_s) & (~hu_s)].sum())                # AI Relevant, Human Not relevant
FN1 = float(w[(~ai_s) & (hu_s)].sum())                # AI Not relevant, Human Relevant
TN1 = float(w[(~ai_s) & (~hu_s)].sum())               # AI Not relevant, Human Not relevant

# Existing metrics
decision_tree_relevance_precision = TP1 / (TP1 + FP1) if (TP1 + FP1) > 0 else np.nan
miss1 = FN1 / (FN1 + TN1) if (FN1 + TN1) > 0 else np.nan

# ----------------------------
# NEW: Cohen's kappa (chance-corrected reliability)
# ----------------------------
kappa1, agreement1, chance_agreement1, weight_total1 = kappa_from_weighted_counts(TP1, FP1, FN1, TN1)

# Optional cross-check using scikit-learn if available; not required for output
try:
    from sklearn.metrics import cohen_kappa_score
    kappa1_sklearn = float(cohen_kappa_score(hu_s.to_numpy(), ai_s.to_numpy(), sample_weight=w.to_numpy()))
except Exception:
    kappa1_sklearn = np.nan

# ----------------------------
# Save outputs
# ----------------------------
summary = pd.DataFrame(
    [{
        "TP1": TP1,
        "FP1": FP1,
        "FN1": FN1,
        "TN1": TN1,
        "decision_tree_relevance_precision": decision_tree_relevance_precision,   # precision of AI "Relevant"
        "miss1": miss1,                                                           # miss rate among AI "Not relevant"
        "kappa1": kappa1,                                                         # Cohen's kappa (weighted)
        "agreement1": agreement1,                                                 # observed agreement (weighted)
        "chance_agreement1": chance_agreement1,                                   # expected by chance (weighted)
        "weight_total1": weight_total1,                                           # sum of weights used
        "n_sample_valid1": int(mask_valid.sum()),                                 # rows used (unsummed)
        "n_sample_flagged1": int(in_sample.sum()),                                # all flagged sample rows
        "n_population1": int(len(df)),                                            # population rows
        "kappa1_sklearn_check": kappa1_sklearn                                    # optional sanity check
    }]
)
summary.to_csv("ai_vs_human_step1.csv", index=False)

conf = pd.DataFrame(
    {
        "Human_Not_relevant": [TN1, FP1],
        "Human_Relevant":     [FN1, TP1],
    },
    index=pd.Index(["AI_Not_relevant", "AI_Relevant"], name="AI")
)
conf.to_csv("confusion_matrix_step1.csv")

print(summary.to_string(index=False))
print("\nConfusion matrix (weighted):")
print(conf)

# Simple confirmation to stdout
if np.isfinite(kappa1) and -1.0 <= kappa1 <= 1.0 and 0.0 <= agreement1 <= 1.0 and 0.0 <= chance_agreement1 <= 1.0:
    print("\n[OK] Cohen's kappa computed and within bounds. agreement1 and chance_agreement1 in [0,1].")
else:
    print("\n[WARN] Kappa or agreements out of bounds or NaN. Check inputs and weights.")
