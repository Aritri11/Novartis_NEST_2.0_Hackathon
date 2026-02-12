# feature_engineering.py
import numpy as np
import pandas as pd
from typing import Dict

# feature_engineering.py (top of file)
import pandas as pd
import numpy as np
from typing import Dict

# columns that uniquely identify a subject across all tables
SUBJECT_KEY = ["study_id", "site_id", "subject_id"]

def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has columns study_id, site_id, subject_id where possible.
    It assumes the ingestion layer already added 'study_id' for each study.
    """
    df = df.copy()
    # study_id should already be there from data_ingestion.load_raw_for_study
    if "study_id" not in df.columns:
        df["study_id"] = np.nan

    # site_id
    if "site_id" not in df.columns:
        for alt in ["site", "site_number", "study_site_number", "siteid", "site_id_"]:
            if alt in df.columns:
                df.rename(columns={alt: "site_id"}, inplace=True)
                break

    # subject_id
    if "subject_id" not in df.columns:
        for alt in ["subject", "subjectname", "subject_id_", "subject_name"]:
            if alt in df.columns:
                df.rename(columns={alt: "subject_id"}, inplace=True)
                break

    return df


# ---------------------------------------------------------------------------
# Aggregators
# ---------------------------------------------------------------------------
def aggregate_visits(visits: pd.DataFrame) -> pd.DataFrame:
    visits = normalize_keys(visits)  # renames site -> site_id, subject -> subject_id
    if visits.empty:
        # return empty frame with expected columns if no data
        return pd.DataFrame(columns=SUBJECT_KEY + ["n_missing_visits", "days_outstanding_max"])

    # visit column
    col_visit = "visit"
    if col_visit not in visits.columns:
        for alt in ["visit_name", "visitname"]:
            if alt in visits.columns:
                col_visit = alt
                break

    # days-outstanding column (based on your printout)
    col_days = "#_days_outstanding"
    if col_days not in visits.columns:
        for alt in ["#_days_outstanding_(today___projected\\ndate)",
                    "#_days_outstanding_(today___projected_date)"]:
            if alt in visits.columns:
                col_days = alt
                break

    # per-subject aggregation: count missing visits and max days outstanding
    grp = visits.groupby(SUBJECT_KEY, as_index=False).agg(
        n_missing_visits=(col_visit, "count"),
        days_outstanding_max=(col_days, "max"),
    )
    return grp


def aggregate_missing_pages(missing_pages: pd.DataFrame) -> pd.DataFrame:
    missing_pages = normalize_keys(missing_pages)
    if missing_pages.empty:
        # no missing-page data -> return schema only
        return pd.DataFrame(columns=SUBJECT_KEY + ["n_missing_pages", "days_page_missing_max"])

    # form / page identifier
    col_form = "formname"
    if col_form not in missing_pages.columns:
        for alt in ["form_name", "page_name", "foldername"]:
            if alt in missing_pages.columns:
                col_form = alt
                break

    # days-missing column – based on your printout
    col_days = "no._#days_page_missing"
    if col_days not in missing_pages.columns:
        for alt in ["#_of_days_missing"]:
            if alt in missing_pages.columns:
                col_days = alt
                break

    # aggregate number of missing pages and worst-case days missing
    grp = missing_pages.groupby(SUBJECT_KEY, as_index=False).agg(
        n_missing_pages=(col_form, "count"),
        days_page_missing_max=(col_days, "max"),
    )
    return grp


def aggregate_lab_issues(missing_lab: pd.DataFrame) -> pd.DataFrame:
    missing_lab = normalize_keys(missing_lab)
    if missing_lab.empty:
        # no lab issues -> only keys with zero count
        return pd.DataFrame(columns=SUBJECT_KEY + ["n_lab_issues"])

    col_issue = "issue_type"
    for alt in ["issue", "issue_description"]:
        if col_issue not in missing_lab.columns and alt in missing_lab.columns:
            col_issue = alt

    # count number of lab issues per subject
    grp = missing_lab.groupby(SUBJECT_KEY, as_index=False).agg(
        n_lab_issues=(col_issue, "count")
    )
    return grp


def aggregate_sae(sae_dm: pd.DataFrame, sae_safety: pd.DataFrame) -> pd.DataFrame:
    dfs = []

    # ---------------- DM sheet ----------------
    if sae_dm is not None and not sae_dm.empty:
        sae_dm = normalize_keys(sae_dm)   # <--- ensure study_id, site_id, subject_id
        sae_dm = sae_dm.copy()

        # subject_id may still be missing if SAE uses patient_id etc.
        if "subject_id" not in sae_dm.columns:
            for alt in ["patient_id", "patient", "subject", "subjectname"]:
                if alt in sae_dm.columns:
                    sae_dm.rename(columns={alt: "subject_id"}, inplace=True)
                    break

        # flag pending actions using action_status when present
        if "action_status" in sae_dm.columns:
            sae_dm["is_pending_action"] = sae_dm["action_status"].str.contains(
                "Pending", case=False, na=False
            )
        else:
            sae_dm["is_pending_action"] = False

        # pick discrepancy / issue identifier column
        disc_col = "discrepancy_id"
        if disc_col not in sae_dm.columns:
            for alt in ["discrepancyid", "disc_id", "issue_id"]:
                if alt in sae_dm.columns:
                    disc_col = alt
                    break

        dfs.append(
            sae_dm.groupby(SUBJECT_KEY, as_index=False).agg(
                n_sae_dm=(disc_col, "nunique"),
                n_sae_dm_pending=("is_pending_action", "sum"),
            )
        )

    # ---------------- Safety sheet ----------------
    if sae_safety is not None and not sae_safety.empty:
        sae_safety = normalize_keys(sae_safety)
        sae_safety = sae_safety.copy()

        if "subject_id" not in sae_safety.columns:
            for alt in ["patient_id", "patient", "subject", "subjectname"]:
                if alt in sae_safety.columns:
                    sae_safety.rename(columns={alt: "subject_id"}, inplace=True)
                    break

        if "action_status" in sae_safety.columns:
            sae_safety["is_pending_action"] = sae_safety["action_status"].str.contains(
                "Pending", case=False, na=False
            )
        else:
            sae_safety["is_pending_action"] = False

        disc_col = "discrepancy_id"
        if disc_col not in sae_safety.columns:
            for alt in ["discrepancyid", "disc_id", "issue_id"]:
                if alt in sae_safety.columns:
                    disc_col = alt
                    break

        dfs.append(
            sae_safety.groupby(SUBJECT_KEY, as_index=False).agg(
                n_sae_safety=(disc_col, "nunique"),
                n_sae_safety_pending=("is_pending_action", "sum"),
            )
        )

    # ---------------- No SAE data ----------------
    if not dfs:
        # schema with zero SAE metrics when no SAE input
        return pd.DataFrame(
            columns=SUBJECT_KEY
            + [
                "n_sae_dm",
                "n_sae_dm_pending",
                "n_sae_safety",
                "n_sae_safety_pending",
                "n_sae_pending_actions",
            ]
        )

    # merge DM and safety aggregates on subject keys
    out = dfs[0]
    for extra in dfs[1:]:
        out = out.merge(extra, on=SUBJECT_KEY, how="outer")

    # fill missing counts with zero
    for c in ["n_sae_dm", "n_sae_dm_pending", "n_sae_safety", "n_sae_safety_pending"]:
        if c not in out.columns:
            out[c] = 0
        out[c] = out[c].fillna(0)

    # total pending actions across DM + safety
    out["n_sae_pending_actions"] = out["n_sae_dm_pending"] + out["n_sae_safety_pending"]
    return out


SUBJECT_KEY = ["study_id", "site_id", "subject_id"]

def _ensure_keys_for_coding(df: pd.DataFrame) -> pd.DataFrame:
    """Force study_id, site_id, subject_id to exist in MedDRA/WHODD tables."""
    df = df.copy()

    # study_id from 'study'
    if "study_id" not in df.columns:
        if "study" in df.columns:
            df.rename(columns={"study": "study_id"}, inplace=True)
        else:
            df["study_id"] = "NA"

    # no site information in these files => dummy 'site_id'
    if "site_id" not in df.columns:
        df["site_id"] = "NA"

    # subject_id from 'subject'
    if "subject_id" not in df.columns:
        if "subject" in df.columns:
            df.rename(columns={"subject": "subject_id"}, inplace=True)
        else:
            df["subject_id"] = "NA"

    return df


def aggregate_coding(medra: pd.DataFrame, whodd: pd.DataFrame) -> pd.DataFrame:
    dfs = []

    # -------- MedDRA --------
    if medra is not None and not medra.empty:
        m = _ensure_keys_for_coding(medra)

        # flag terms that require coding
        if "require_coding" in m.columns:
            m["requires_coding"] = m["require_coding"].astype(str).str.upper().eq("YES")
        else:
            m["requires_coding"] = True

        # identify uncoded terms
        if "coding_status" in m.columns:
            m["is_uncoded"] = m["coding_status"].str.contains("UnCoded", case=False, na=False)
        else:
            m["is_uncoded"] = False

        log_col = "logline"  # present in your columns
        dfs.append(
            m.groupby(SUBJECT_KEY, as_index=False).agg(
                n_medra_terms=(log_col, "count"),
                n_medra_uncoded=("is_uncoded", "sum"),
                n_medra_requires_coding=("requires_coding", "sum"),
            )
        )

    # -------- WHODD / WHO Drug --------
    if whodd is not None and not whodd.empty:
        w = _ensure_keys_for_coding(whodd)

        if "require_coding" in w.columns:
            w["requires_coding"] = w["require_coding"].astype(str).str.upper().eq("YES")
        else:
            w["requires_coding"] = True

        if "coding_status" in w.columns:
            w["is_uncoded"] = w["coding_status"].str.contains("UnCoded", case=False, na=False)
        else:
            w["is_uncoded"] = False

        log_col = "logline"
        dfs.append(
            w.groupby(SUBJECT_KEY, as_index=False).agg(
                n_whodd_terms=(log_col, "count"),
                n_whodd_uncoded=("is_uncoded", "sum"),
                n_whodd_requires_coding=("requires_coding", "sum"),
            )
        )

    # -------- No coding data --------
    if not dfs:
        # return template shape when there is no MedDRA/WHODD data at all
        return pd.DataFrame(
            columns=SUBJECT_KEY
            + [
                "n_medra_terms",
                "n_medra_uncoded",
                "n_medra_requires_coding",
                "n_whodd_terms",
                "n_whodd_uncoded",
                "n_whodd_requires_coding",
                "n_uncoded_terms",
                "n_terms_requires_coding",
            ]
        )

    # merge MedDRA and WHODD metrics
    out = dfs[0]
    for extra in dfs[1:]:
        out = out.merge(extra, on=SUBJECT_KEY, how="outer")

    # fill missing coding counts with zero
    for c in [
        "n_medra_terms",
        "n_medra_uncoded",
        "n_medra_requires_coding",
        "n_whodd_terms",
        "n_whodd_uncoded",
        "n_whodd_requires_coding",
    ]:
        if c not in out.columns:
            out[c] = 0
        out[c] = out[c].fillna(0)

    # aggregate total uncoded terms and total requiring coding
    out["n_uncoded_terms"] = out["n_medra_uncoded"] + out["n_whodd_uncoded"]
    out["n_terms_requires_coding"] = (
        out["n_medra_requires_coding"] + out["n_whodd_requires_coding"]
    )
    return out


def aggregate_edrr(edrr: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates Compiled EDRR unresolved-issue counts per subject.
    Ensures study_id, site_id, subject_id exist for grouping.
    """
    if edrr is None or edrr.empty:
        # no EDRR issues -> zero count
        return pd.DataFrame(columns=SUBJECT_KEY + ["n_open_edrr_issues"])

    edrr = edrr.copy()

    # ---- study_id ----
    if "study_id" not in edrr.columns:
        if "study" in edrr.columns:
            edrr.rename(columns={"study": "study_id"}, inplace=True)
        elif "study_name" in edrr.columns:
            edrr.rename(columns={"study_name": "study_id"}, inplace=True)
        else:
            edrr["study_id"] = "NA"

    # ---- site_id ----
    if "site_id" not in edrr.columns:
        for alt in ["site", "site_number", "sitenumber", "study_site_number", "siteid"]:
            if alt in edrr.columns:
                edrr.rename(columns={alt: "site_id"}, inplace=True)
                break
        else:
            # if the file has no site information at all, use a dummy
            edrr["site_id"] = "NA"

    # ---- subject_id ----
    if "subject_id" not in edrr.columns:
        for alt in ["subject", "subjectname", "subject_name", "patient_id", "patient"]:
            if alt in edrr.columns:
                edrr.rename(columns={alt: "subject_id"}, inplace=True)
                break
        else:
            edrr["subject_id"] = "NA"

    # ---- issue-count column ----
    col_issues = "total_open_issue_count_per_subject"
    if col_issues not in edrr.columns:
        numeric_cols = edrr.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            col_issues = numeric_cols[0]
        else:
            # no numeric column -> treat as 0 issues
            edrr[col_issues] = 0

    # sum total open issues per subject
    grp = edrr.groupby(SUBJECT_KEY, as_index=False).agg(
        n_open_edrr_issues=(col_issues, "sum")
    )
    return grp


# ---------------------------------------------------------------------------
# CPID features
# ---------------------------------------------------------------------------

def engineer_from_cpid(cpid: pd.DataFrame) -> pd.DataFrame:
    df = normalize_keys(cpid)
    df = cpid.copy()

    # Ensure keys exist; your CPID files use e.g. "site_id"/"subject_id"
    # If names differ, change here once.
    if "site_id" not in df.columns:
        # common alternative: "site" or "site_number"
        for alt in ["site", "site_number", "study_site_number"]:
            if alt in df.columns:
                df.rename(columns={alt: "site_id"}, inplace=True)
                break

    if "subject_id" not in df.columns:
        for alt in ["subject", "subjectname"]:
            if alt in df.columns:
                df.rename(columns={alt: "subject_id"}, inplace=True)
                break

    # Query columns (adapt if your exact names differ)
    rename_map = {
        "dm_queries": "n_dm_queries",
        "clinical_queries": "n_clinical_queries",
        "medical_queries": "n_medical_queries",
        "site_queries": "n_site_queries",
        "field_monitor_queries": "n_field_monitor_queries",
        "coding_queries": "n_coding_queries",
        "safety_queries": "n_safety_queries",
        "total_queries": "n_total_queries",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # derive total queries if not already present
    if "n_total_queries" not in df.columns:
        q_cols = [c for c in df.columns if c.endswith("_queries")]
        if q_cols:
            df["n_total_queries"] = df[q_cols].sum(axis=1)
        else:
            df["n_total_queries"] = 0

    # Nonconformance / CRF counts (adjust to your CPID columns)
    n_crf_col = "pages_entered" if "pages_entered" in df.columns else None
    if n_crf_col is None:
        # fallback: choose first numeric column that looks like count
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            n_crf_col = num_cols[0]

    if n_crf_col is None:
        df["n_crfs_total"] = 1
    else:
        df["n_crfs_total"] = df[n_crf_col].replace(0, np.nan)

    if "pages_with_non_conformant_data" in df.columns:
        df["n_nonconformant_pages"] = df["pages_with_non_conformant_data"]
    else:
        df["n_nonconformant_pages"] = 0

    # Percentages
    df["pct_crfs_with_nonconformance"] = (
        df["n_nonconformant_pages"] / df["n_crfs_total"]
    )

    if "forms_verified" in df.columns:
        df["pct_crfs_verified"] = df["forms_verified"] / df["n_crfs_total"]
    else:
        df["pct_crfs_verified"] = 0

    if "crfs_signed" in df.columns:
        df["pct_crfs_signed"] = df["crfs_signed"] / df["n_crfs_total"]
    else:
        df["pct_crfs_signed"] = 0

    overdue_cols = [
        c for c in df.columns
        if "overdue_for_signs" in c or "overdue" in c and "sign" in c
    ]
    if overdue_cols:
        df["pct_crfs_overdue"] = df[overdue_cols].sum(axis=1) / df["n_crfs_total"]
    else:
        df["pct_crfs_overdue"] = 0

    # Open queries: if explicit column not available, approximate with total
    if "open_queries" in df.columns:
        df["n_open_queries"] = df["open_queries"]
    else:
        df["n_open_queries"] = df["n_total_queries"]

    return df

# ---------------------------------------------------------------------------
# Build subject snapshot for ALL studies
# ---------------------------------------------------------------------------

def build_subject_snapshot(raw: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    # base CPID features (one row per subject)
    cpid_feat = engineer_from_cpid(raw["cpid"])

    # aggregate each auxiliary table to subject level
    visits_feat = aggregate_visits(raw.get("visits", pd.DataFrame()))
    missing_pages_feat = aggregate_missing_pages(raw.get("missing_pages", pd.DataFrame()))
    lab_feat = aggregate_lab_issues(raw.get("missing_lab", pd.DataFrame()))
    sae_feat = aggregate_sae(
        raw.get("sae_dm", pd.DataFrame()),
        raw.get("sae_safety", pd.DataFrame()),
    )
    coding_feat = aggregate_coding(
        raw.get("medra", pd.DataFrame()),
        raw.get("whodd", pd.DataFrame()),
    )
    edrr_feat = aggregate_edrr(raw.get("edrr", pd.DataFrame()))

    # merge everything left-join style onto CPID subject universe
    df = cpid_feat.copy()
    for feat in [visits_feat, missing_pages_feat, lab_feat, sae_feat, coding_feat, edrr_feat]:
        if not feat.empty:
            df = df.merge(feat, on=SUBJECT_KEY, how="left")

    # replace NaNs in numeric columns with zero for downstream scoring
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)
    return df
