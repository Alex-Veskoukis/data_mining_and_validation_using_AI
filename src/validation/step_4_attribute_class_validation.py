"""
Implements audit Step 4 from "Governance by Evidence: Regulated Predictors in
Decision-Tree Models" (Section 3.2): assess regulated data category (RDC)
labeling on audited features that passed upstream AI gates. Applying audit
weights, this code computes the weighted match rate and Cohen's kappa across
attribute classes that are reported in the paper and yield the matchRDC (A4)
factor used in the propagated multiplier S (Section 3.3).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.io import PROC

PROCESSED_DIR = Path(PROC)
INPUT_PATH = PROCESSED_DIR / "attribute_classes_audited.json"

STRATA_COLS = ["source", "domain"]
SAMPLE_FLAG_COL = "feature_validation_audited"

def load_json_df(path: Path) -> pd.DataFrame:
    try:
        return pd.read_json(path, lines=True)
    except ValueError:
        return pd.read_json(path)

def require_cols(df: pd.DataFrame, cols):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}")

def norm_class(series: pd.Series) -> pd.Series:
    s = series.astype("string")
    s = s.str.strip().str.replace(r"\s+", "_", regex=True).str.replace("-", "_")
    s = s.str.replace("__", "_")
    return s

def norm_valid(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip().str.lower()
    out = pd.Series(np.nan, index=series.index, dtype="float")
    out[s == "valid"] = True
    out[s == "not valid"] = False
    return out

def compute_audit_weights(df: pd.DataFrame,
                          sample_mask: pd.Series,
                          strata_cols: list) -> pd.Series:
    w = pd.Series(np.nan, index=df.index, dtype=float)
    Npop = len(df)
    Nsam = int(sample_mask.sum())
    if Nsam == 0:
        raise ValueError("No audited features flagged; check 'feature_validation_audited'.")
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

def kappa_nominal_from_labels(ai_labels: pd.Series, hu_labels: pd.Series, weights: pd.Series):
    """
    Cohen's kappa for nominal multi-class with sample weights.
    Returns (kappa, observed_agreement, chance_agreement, W, n_pairs).
    """
    tab = pd.crosstab(ai_labels, hu_labels, values=weights, aggfunc="sum", dropna=False).fillna(0.0)
    labels = sorted(set(tab.index.tolist()) | set(tab.columns.tolist()))
    tab = tab.reindex(index=labels, columns=labels, fill_value=0.0).to_numpy(dtype=float)

    W = tab.sum()
    if not np.isfinite(W) or W <= 0:
        return np.nan, np.nan, np.nan, 0.0, 0

    po = np.trace(tab) / W
    r = tab.sum(axis=1)
    c = tab.sum(axis=0)
    pe = float(np.dot(r, c)) / float(W * W)

    denom = 1.0 - pe
    if denom <= 0:
        return np.nan, po, pe, W, int(len(weights))
    kappa = (po - pe) / denom
    return float(kappa), float(po), float(pe), float(W), int(len(weights))

# -------- Load
df = load_json_df(INPUT_PATH)
require_cols(df, ["doi", SAMPLE_FLAG_COL, "attribute_class", "attribute_class_human_verified"])

# -------- AI-only gates (pass if missing)
in_sample = df[SAMPLE_FLAG_COL] == 1
gate_rel = df.get("decision_trees_related", "").astype("string").str.strip().str.lower().eq("relevant") \
           if "decision_trees_related" in df.columns else pd.Series(True, index=df.index)
gate_dom = (df["domain_validated"].astype("string").str.strip().str.lower()
            .eq(df["domain"].astype("string").str.strip().str.lower())) \
           if {"domain_validated","domain"} <= set(df.columns) else pd.Series(True, index=df.index)
gate_feat = norm_valid(df["feature_validation"]).fillna(False) \
           if "feature_validation" in df.columns else pd.Series(True, index=df.index)

mask_eval = in_sample & gate_rel & gate_dom & gate_feat

# -------- Weights and labels
audit_weight = compute_audit_weights(df, in_sample, STRATA_COLS)
ai_class = norm_class(df["attribute_class"])
hu_class = norm_class(df["attribute_class_human_verified"])

m = mask_eval & ai_class.notna() & hu_class.notna()
if not m.any():
    raise ValueError("No audited feature rows with both AI and human classes after AI gates.")

w  = audit_weight[m].astype(float)
ai = ai_class[m]
hu = hu_class[m]

# -------- Match rate (existing)
matches = float(w[(ai == hu)].sum())
total_w = float(w.sum())
mismatches = total_w - matches
attribute_class_precision = matches / total_w if total_w > 0 else np.nan

# -------- NEW: Cohen's kappa
kappa4, agreement4, chance4, W4, n_pairs4 = kappa_nominal_from_labels(ai, hu, w)

# Optional cross-check if sklearn is present
try:
    from sklearn.metrics import cohen_kappa_score
    kappa4_sklearn = float(cohen_kappa_score(hu, ai, sample_weight=w))
except Exception:
    kappa4_sklearn = np.nan

# -------- Save summary
pd.DataFrame([{
    "attribute_class_precision": attribute_class_precision,
    "matches": matches,
    "mismatches": mismatches,
    "n_features_audited": total_w,
    "kappa4": kappa4,
    "agreement4": agreement4,
    "chance_agreement4": chance4,
    "weight_total4": W4,
    "n_valid_pairs4": n_pairs4,
    "kappa4_sklearn_check": kappa4_sklearn
}]).to_csv("ai_vs_human_step4.csv", index=False)

# -------- Confusion and per-class shares (unchanged)
conf = (
    pd.DataFrame({"w": w, "AI": ai, "Human": hu})
      .pivot_table(index="AI", columns="Human", values="w", aggfunc="sum", fill_value=0.0)
      .sort_index(axis=0).sort_index(axis=1)
)
conf.to_csv("aclass_confusion.csv")

df_by = pd.DataFrame({"w": w, "AI": ai, "Human": hu})
grp = df_by.groupby("Human", observed=False, dropna=False)
tot = grp["w"].sum().rename("w_total")
hit = grp.apply(lambda g: float(g.loc[g["AI"] == g.name, "w"].sum())).rename("w_match")
out = pd.concat([tot, hit], axis=1)
out["share_ai_equals_human"] = out["w_match"] / out["w_total"]
out.reset_index().rename(columns={"Human": "human_class"}).to_csv("aclass_match_by_class.csv", index=False)

# Console confirmation
print(pd.read_csv("ai_vs_human_step4.csv").to_string(index=False))
if np.isfinite(kappa4) and -1.0 <= kappa4 <= 1.0 and 0.0 <= agreement4 <= 1.0 and 0.0 <= chance4 <= 1.0:
    print("\n[OK] Cohen's kappa computed and within bounds.")
else:
    print("\n[WARN] Check kappa bounds.")
