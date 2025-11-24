"""
Implements audit Step 2 from "Governance by Evidence: Regulated Predictors in
Decision-Tree Models" (Section 3.2): evaluate the industry/domain assignment
after the relevance gate on the same stratified 1,000-DOI audit sample.
Weighting Crossref/OpenAlex rows, this code computes gate-pass purity
(domain_precision, pass_wrong) and Cohen's kappa both overall and on gate==1,
which are reported in the paper and provide the A2 factor in the propagated
multiplier S (Section 3.3).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.io import PROC

PROCESSED_DIR = Path(PROC)
INPUT_PATH = PROCESSED_DIR / "merged_domain_validated_audited.json"

STRATA_COLS = ["source", "domain"]       # set [] if none
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

def norm_yes(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip().str.lower()
    return s.isin({"relevant","true","1","yes","y"})

def norm_dom(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip().str.lower()
    return (
        s.str.replace(r"\s+", "_", regex=True)
         .str.replace("-", "_")
         .str.replace("__", "_")
    )

def compute_audit_weights(df: pd.DataFrame, sample_mask: pd.Series, strata_cols: list) -> pd.Series:
    w = pd.Series(np.nan, index=df.index, dtype=float)
    Npop, Nsam = len(df), int(sample_mask.sum())
    if Nsam == 0:
        raise ValueError("Sample size is zero; check 'desicion_tree_relevant_audited'.")
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
    Cohen's kappa for nominal categories from weighted crosstab.
    Returns (kappa, observed_agreement, chance_agreement, W, n_rows).
    """
    # Build weighted crosstab
    tab = pd.crosstab(
        ai_labels, hu_labels, values=weights, aggfunc="sum", dropna=False
    ).fillna(0.0)

    # Align to union of labels for a square matrix
    labels = sorted(set(tab.index.tolist()) | set(tab.columns.tolist()))
    tab = tab.reindex(index=labels, columns=labels, fill_value=0.0)

    M = tab.to_numpy(dtype=float)
    W = M.sum()
    if not np.isfinite(W) or W <= 0:
        return np.nan, np.nan, np.nan, 0.0, 0

    po = np.trace(M) / W
    r = M.sum(axis=1)   # AI marginals
    c = M.sum(axis=0)   # Human marginals
    pe = float(np.dot(r, c)) / float(W * W)

    denom = 1.0 - pe
    if denom <= 0:
        return np.nan, po, pe, W, int(len(weights))
    kappa = (po - pe) / denom
    return float(kappa), float(po), float(pe), float(W), int(len(weights))

# ---- Load
df = load_json_df(INPUT_PATH)
require_cols(df, [
    SAMPLE_FLAG_COL,
    "decision_trees_related",
    "domain",
    "domain_validated",
    "domain_validated_human_verified",
])

# ---- Masks, weights, normalized fields
in_sample   = df[SAMPLE_FLAG_COL] == 1
ai_relevant = norm_yes(df["decision_trees_related"])
ai_domain   = norm_dom(df["domain_validated"])
human_dom   = norm_dom(df["domain_validated_human_verified"])
searched    = norm_dom(df["domain"])

w = compute_audit_weights(df, in_sample, STRATA_COLS)

# Audited rows for domain step: sampled AND AI Relevant
mask_domain = in_sample & ai_relevant

# AI gate and human agreement vs searched domain
ai_gate = (ai_domain == searched)
hu_eq   = (human_dom == searched)

# ----------------------------
# Purity-focused metrics on gate==1 only
# ----------------------------
m_pass = mask_domain & ai_gate & human_dom.notna() & searched.notna()
w_pass_total = float(w[m_pass].sum())
w_pass_ok    = float(w[m_pass & hu_eq].sum())
w_pass_wrong = float(w_pass_total - w_pass_ok)

domain_precision = (w_pass_ok / w_pass_total) if w_pass_total > 0 else np.nan
pass_wrong       = (w_pass_wrong / w_pass_total) if w_pass_total > 0 else np.nan

# ----------------------------
# NEW: Cohen's kappa (nominal)
#  A) All audited rows with both labels present
#  B) Gate==1 subset (optional, reported for completeness)
# ----------------------------
m_all = mask_domain & ai_domain.notna() & human_dom.notna()

kappa_all, agree_all, chance_all, W_all, n_all = kappa_nominal_from_labels(
    ai_domain[m_all], human_dom[m_all], w[m_all].astype(float)
)

m_gate = m_all & ai_gate
kappa_gate, agree_gate, chance_gate, W_gate, n_gate = kappa_nominal_from_labels(
    ai_domain[m_gate], human_dom[m_gate], w[m_gate].astype(float)
)

# Optional cross-check via scikit-learn if available
try:
    from sklearn.metrics import cohen_kappa_score
    kappa_all_sklearn = float(cohen_kappa_score(human_dom[m_all], ai_domain[m_all], sample_weight=w[m_all]))
    kappa_gate_sklearn = float(cohen_kappa_score(human_dom[m_gate], ai_domain[m_gate], sample_weight=w[m_gate]))
except Exception:
    kappa_all_sklearn = np.nan
    kappa_gate_sklearn = np.nan

# ----------------------------
# Save outputs
# ----------------------------
out = pd.DataFrame([{
    # Existing purity metrics (gate==1)
    "domain_precision": domain_precision,
    "pass_wrong": pass_wrong,
    "w_gate1_total": w_pass_total,

    # NEW reliability metrics
    "kappa_domain_all": kappa_all,
    "agreement_domain_all": agree_all,
    "chance_agreement_domain_all": chance_all,
    "weight_total_domain_all": W_all,
    "n_valid_domain_all": n_all,

    "kappa_domain_gate1": kappa_gate,
    "agreement_domain_gate1": agree_gate,
    "chance_agreement_domain_gate1": chance_gate,
    "weight_total_domain_gate1": W_gate,
    "n_valid_domain_gate1": n_gate,

    # Optional checks
    "kappa_all_sklearn_check": kappa_all_sklearn,
    "kappa_gate_sklearn_check": kappa_gate_sklearn,
}])
out.to_csv("ai_vs_human_step2.csv", index=False)

# Keep your diagnostic confusion as well
m_conf = m_all
conf = (
    pd.DataFrame({"w": w[m_conf].astype(float), "AI": ai_domain[m_conf], "Human": human_dom[m_conf]})
      .pivot_table(index="AI", columns="Human", values="w", aggfunc="sum", fill_value=0.0)
      .sort_index(axis=0).sort_index(axis=1)
)
conf.to_csv("domain_confusion.csv")

print(out.to_string(index=False))

# Simple confirmation to stdout
ok_bounds = (
    (np.isnan(kappa_all) or -1.0 <= kappa_all <= 1.0) and
    (np.isnan(kappa_gate) or -1.0 <= kappa_gate <= 1.0) and
    (np.isnan(agree_all) or 0.0 <= agree_all <= 1.0) and
    (np.isnan(chance_all) or 0.0 <= chance_all <= 1.0) and
    (np.isnan(agree_gate) or 0.0 <= agree_gate <= 1.0) and
    (np.isnan(chance_gate) or 0.0 <= chance_gate <= 1.0)
)
print("\n[OK] Kappa computed and within bounds." if ok_bounds else "\n[WARN] Check kappa bounds.")
