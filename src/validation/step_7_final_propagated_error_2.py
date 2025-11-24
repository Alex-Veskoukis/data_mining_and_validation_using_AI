"""
Implements status-only diagnostics from "Governance by Evidence: Regulated
Predictors in Decision-Tree Models" (Section 3.2): on the audited, AI-gated
pairs, measure how often AI "Regulated" (any) or "Regulated+High" status calls
are confirmed by humans. Uses the same stratified weights to report purity for
those final sets plus Cohen's kappa on status-only labels, matching the
regulation-status audit and per-regulation purity tables reported in the paper.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.io import PROC

PROCESSED_DIR = Path(PROC)
PATH = PROCESSED_DIR / "validated_feature_regulation_audited.json"

STRATA_COLS = ["source", "domain"]
SAMPLE_FLAG_FALLBACKS = ["feature_validation_audited", "desicion_tree_relevant_audited"]

def load_json_df(p: Path) -> pd.DataFrame:
    try:
        return pd.read_json(p, lines=True)
    except ValueError:
        return pd.read_json(p)

def pick_sample_flag(df: pd.DataFrame) -> str:
    for c in SAMPLE_FLAG_FALLBACKS:
        if c in df.columns:
            return c
    raise ValueError("Sample flag not found. Expected one of: " + ", ".join(SAMPLE_FLAG_FALLBACKS))

def compute_audit_weights(df, sample_mask, strata_cols):
    w = pd.Series(np.nan, index=df.index, dtype=float)
    Npop = len(df)
    Nsam = int(sample_mask.sum())
    if Nsam == 0:
        raise ValueError("No audited rows found for weighting.")
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

def gate_bool(df, col, true_val):
    if col not in df.columns:
        return pd.Series(True, index=df.index)
    s = df[col].astype("string").str.strip().str.lower()
    return s.eq(true_val)

def gate_equal(df, col_a, col_b):
    if not ({col_a, col_b} <= set(df.columns)):
        return pd.Series(True, index=df.index)
    a = df[col_a].astype("string").str.strip().str.lower()
    b = df[col_b].astype("string").str.strip().str.lower()
    return a.eq(b)

def ai_reg_high(df):
    r = df["regulation_status"].astype("string").str.strip().str.lower()
    c = df["confidence"].astype("string").str.strip().str.lower()
    return r.eq("regulated") & c.eq("high")

def ai_reg_any(df):
    r = df["regulation_status"].astype("string").str.strip().str.lower()
    return r.eq("regulated")

def human_reg_any(df):
    if ("regulation_status_human_verified" not in df.columns):
        return pd.Series(False, index=df.index)
    r = df["regulation_status_human_verified"].astype("string").str.strip().str.lower()
    return r.eq("regulated")

def norm_status(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip().str.lower()
    out = pd.Series(pd.NA, index=series.index, dtype="string")
    out[s.eq("regulated")] = "regulated"
    out[s.eq("not regulated")] = "not_regulated"
    return out

def kappa_from_labels(ai_labels: pd.Series, hu_labels: pd.Series, weights: pd.Series):
    """Cohen's kappa for nominal 2-class using weighted contingency."""
    tab = pd.crosstab(ai_labels, hu_labels, values=weights, aggfunc="sum", dropna=False).fillna(0.0)
    labels = sorted(set(tab.index.tolist()) | set(tab.columns.tolist()))
    tab = tab.reindex(index=labels, columns=labels, fill_value=0.0).to_numpy(dtype=float)
    W = tab.sum()
    if not np.isfinite(W) or W <= 0:
        return np.nan, np.nan, np.nan, 0.0, 0
    po = np.trace(tab) / W
    r  = tab.sum(axis=1)
    c  = tab.sum(axis=0)
    pe = float(np.dot(r, c)) / float(W * W)
    denom = 1.0 - pe
    if denom <= 0:
        return np.nan, po, pe, W, int(len(weights))
    kappa = (po - pe) / denom
    return float(kappa), float(po), float(pe), float(W), int(len(weights))

# ---- load
df = load_json_df(PATH)
needed = ["regulation_status","confidence","reg_id","regulation_status_human_verified"]
missing = [c for c in needed if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

sample_flag = pick_sample_flag(df)
in_sample = df[sample_flag] == 1
w = compute_audit_weights(df, in_sample, STRATA_COLS)

# AI-only gates (pass if missing)
g_rel  = gate_bool(df, "decision_trees_related", "relevant")
g_dom  = gate_equal(df, "domain_validated", "domain")
g_feat = gate_bool(df, "feature_validation", "valid")
mask_eval = in_sample & g_rel & g_dom & g_feat

# Booleans and labels
ai_high = ai_reg_high(df)
ai_any  = ai_reg_any(df)
hu_any  = human_reg_any(df)
has_human = df["regulation_status_human_verified"].notna()

# Evaluate sets
mA = mask_eval & has_human & ai_high  # AI RegHigh rows with human labels
mB = mask_eval & has_human & ai_any   # AI RegAny  rows with human labels

# Purities
denA = float(w[mA].sum()); numA = float(w[mA & hu_any].sum())
denB = float(w[mB].sum()); numB = float(w[mB & hu_any].sum())
purity_A = numA / denA if denA > 0 else np.nan
purity_B = numB / denB if denB > 0 else np.nan

# NEW: Cohen's kappa on status-only within each evaluated set
ai_status = norm_status(df["regulation_status"])
hu_status = norm_status(df["regulation_status_human_verified"])

kA, PoA, PeA, WA, nA = kappa_from_labels(ai_status[mA], hu_status[mA], w[mA])
kB, PoB, PeB, WB, nB = kappa_from_labels(ai_status[mB], hu_status[mB], w[mB])

# Optional sklearn cross-checks
try:
    from sklearn.metrics import cohen_kappa_score
    kA_sklearn = float(cohen_kappa_score(hu_status[mA], ai_status[mA], sample_weight=w[mA]))
    kB_sklearn = float(cohen_kappa_score(hu_status[mB], ai_status[mB], sample_weight=w[mB]))
except Exception:
    kA_sklearn = np.nan
    kB_sklearn = np.nan

pd.DataFrame([{
    "purity_status_only_AI_RegHigh": purity_A,
    "den_weight_RegHigh": denA,
    "kappa_status_only_AI_RegHigh": kA,
    "agreement_status_only_AI_RegHigh": PoA,
    "chance_agreement_status_only_AI_RegHigh": PeA,
    "weight_total_AI_RegHigh": WA,
    "n_pairs_AI_RegHigh": nA,
    "kappa_status_only_AI_RegHigh_sklearn_check": kA_sklearn,

    "purity_status_only_AI_RegAny":  purity_B,
    "den_weight_RegAny":  denB,
    "kappa_status_only_AI_RegAny": kB,
    "agreement_status_only_AI_RegAny": PoB,
    "chance_agreement_status_only_AI_RegAny": PeB,
    "weight_total_AI_RegAny": WB,
    "n_pairs_AI_RegAny": nB,
    "kappa_status_only_AI_RegAny_sklearn_check": kB_sklearn,
}]).to_csv("final_purity_status_only.csv", index=False)

print(pd.read_csv("final_purity_status_only.csv").to_string(index=False))

# Per-reg tables unchanged
def per_reg(df_mask):
    dfr = pd.DataFrame({
        "reg": df.loc[df_mask, "reg_id"].astype("string").str.strip(),
        "w":   w[df_mask].astype(float),
        "huR": hu_any[df_mask].astype(bool)
    })
    den = dfr.groupby("reg", observed=False, dropna=False)["w"].sum()
    num = dfr.loc[dfr["huR"]].groupby("reg", observed=False, dropna=False)["w"].sum()
    out = pd.DataFrame({"den": den}).join(num.rename("num")).fillna(0.0)
    out["purity"] = np.where(out["den"] > 0, out["num"]/out["den"], np.nan)
    return out.reset_index().rename(columns={"reg":"reg_id"})

per_reg(mA).to_csv("final_purity_status_only_by_reg_AI_RegHigh.csv", index=False)
per_reg(mB).to_csv("final_purity_status_only_by_reg_AI_RegAny.csv", index=False)
