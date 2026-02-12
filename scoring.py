# scoring.py
import numpy as np
import pandas as pd

def compute_clean_patient_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rule-based 'clean patient' definition across all studies.
    Ensures all required driver columns exist; missing ones default to 0.
    """
    d = df.copy()

    # ensure all numeric driver columns exist
    required_zero_cols = [
        "n_missing_visits",
        "n_missing_pages",
        "n_open_queries",
        "n_nonconformant_pages",
        "n_lab_issues",
        "n_uncoded_terms",
        "n_open_edrr_issues",
        "n_sae_pending_actions",
    ]
    for c in required_zero_cols:
        if c not in d.columns:
            d[c] = 0

    # and percentage CRF columns
    pct_cols = ["pct_crfs_verified", "pct_crfs_signed", "pct_crfs_overdue"]
    for c in pct_cols:
        if c not in d.columns:
            # default: 0 verified/signed, 0 overdue
            d[c] = 0.0

    d = d.fillna(0).infer_objects(copy=False)

    conds = [
        d["n_missing_visits"].eq(0),
        d["n_missing_pages"].eq(0),
        d["n_open_queries"].eq(0),
        d["n_nonconformant_pages"].eq(0),
        d["n_lab_issues"].eq(0),
        d["n_uncoded_terms"].eq(0),
        d["n_open_edrr_issues"].eq(0),
        d["n_sae_pending_actions"].eq(0),
        d["pct_crfs_verified"].ge(1.0),
        d["pct_crfs_signed"].ge(1.0),
        d["pct_crfs_overdue"].eq(0),
    ]
    d["clean_patient"] = np.logical_and.reduce(conds).astype(int)
    return d



def _bounded_inverse_rate(value, threshold):
    val = value / threshold
    val = np.clip(val, 0, 1)
    return 1 - val


def compute_dqi(df: pd.DataFrame,
                t_missing=3,
                t_queries=10,
                t_lab=3,
                t_coding=5,
                t_safety=1) -> pd.DataFrame:

    d = df.copy()

    d["s_missing"] = _bounded_inverse_rate(
        d.get("n_missing_visits", 0) + d.get("n_missing_pages", 0), t_missing
    )
    d["s_queries"] = _bounded_inverse_rate(d.get("n_open_queries", 0), t_queries)
    d["s_verification"] = (
        0.4 * d.get("pct_crfs_signed", 0).fillna(0) +
        0.4 * d.get("pct_crfs_verified", 0).fillna(0) +
        0.2 * (1 - d.get("pct_crfs_overdue", 0).fillna(0))
    )
    d["s_lab"] = _bounded_inverse_rate(d.get("n_lab_issues", 0), t_lab)
    d["s_coding"] = _bounded_inverse_rate(d.get("n_uncoded_terms", 0), t_coding)
    d["s_safety"] = _bounded_inverse_rate(d.get("n_sae_pending_actions", 0), t_safety)

    d["dqi"] = (
        0.15 * d["s_missing"] +
        0.20 * d["s_queries"] +
        0.20 * d["s_verification"] +
        0.10 * d["s_lab"] +
        0.10 * d["s_coding"] +
        0.25 * d["s_safety"]
    )

    d["dqi_band"] = pd.cut(
        d["dqi"],
        bins=[-0.01, 0.6, 0.85, 1.0],
        labels=["Red", "Amber", "Green"]
    )

    return d
