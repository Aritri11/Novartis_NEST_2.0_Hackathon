# ai_utils.py
from typing import Dict, Any

import pandas as pd
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage

load_dotenv()

# base LLM endpoint (Zephyr 7B chat model hosted on Hugging Face Inference)
_llm=HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task= "text-generation"
)

# LangChain chat wrapper around the endpoint
model= ChatHuggingFace(llm=_llm)


def _llm_call(prompt: str) -> str:
    """Thin wrapper to call the LangChain HuggingFace chat model."""
    msg = HumanMessage(content=prompt)
    # ChatHuggingFace is a Runnable; use .invoke, not model([...])
    resp = model.invoke([msg])
    return resp.content.strip()

# ---------------------------------------------------------------------
# 1) Study-level narrative summary
# ---------------------------------------------------------------------

def summarize_study(df_study: pd.DataFrame, study_id: str) -> str:
    """
    Generate a concise, action-oriented narrative for the selected study.
    """
    if df_study.empty:
        return f"Study {study_id}: no subjects available for summary."

    # high-level study metrics to feed into the prompt
    n_subj = len(df_study)
    mean_dqi = float(df_study["dqi"].mean())
    pct_red = float((df_study["dqi_band"] == "Red").mean() * 100)
    pct_clean = float(df_study["clean_patient"].mean() * 100)

    # roll up site-level metrics and sort so worst sites appear first
    site_rollup = (
        df_study.groupby("site_id", as_index=False)
        .agg(
            mean_dqi=("dqi", "mean"),
            pct_clean=("clean_patient", "mean"),
            n_subjects=("subject_id", "nunique"),
            n_red=("dqi_band", lambda x: (x == "Red").sum()),
        )
        .sort_values("mean_dqi")
    )

    # keep just a small JSON context for the model
    worst_sites: Dict[str, Any] = (
        site_rollup.head(5)
        .to_dict(orient="records")
    )

    # instruction-style prompt guiding the LLM to a short, practical summary
    prompt = f"""
You are an expert clinical operations assistant.

You have summary metrics for Study {study_id}:

- Number of subjects: {n_subj}
- Mean Data Quality Index (DQI, 0–1): {mean_dqi:.3f}
- Percent clean patients: {pct_clean:.1f}%
- Percent Red-band patients: {pct_red:.1f}%

You also have the 5 lowest-DQI sites (JSON list of dicts):
{worst_sites}

Write a SHORT, action-oriented narrative for the study team:

- Start with 1 sentence giving overall data-quality status.
- Then give 4–6 bullet points.
- Highlight which sites or patterns are most concerning.
- Propose concrete operational actions (e.g., query resolution, SDV focus, lab reconciliation, coding clean-up).
- Use clear language (no equations, no code).
"""
    return _llm_call(prompt)


# ---------------------------------------------------------------------
# 2) Site-level operational recommendations
# ---------------------------------------------------------------------

def recommend_actions_for_site(df_site: pd.DataFrame, study_id: str, site_id: str) -> str:
    """
    Generate context-aware recommendations for a single site using LangChain.
    """
    if df_site.empty:
        return f"No subjects for Study {study_id}, Site {site_id}."

    # basic site summary used in the prompt
    n_subj = len(df_site)
    mean_dqi = float(df_site["dqi"].mean())
    pct_red = float((df_site["dqi_band"] == "Red").mean() * 100)

    # aggregate issue counts across all subjects at the site
    totals = {
        "n_missing_visits": int(df_site.get("n_missing_visits", 0).sum()),
        "n_missing_pages": int(df_site.get("n_missing_pages", 0).sum()),
        "n_open_queries": int(df_site.get("n_open_queries", 0).sum()),
        "n_lab_issues": int(df_site.get("n_lab_issues", 0).sum()),
        "n_uncoded_terms": int(df_site.get("n_uncoded_terms", 0).sum()),
        "n_open_edrr_issues": int(df_site.get("n_open_edrr_issues", 0).sum()),
        "n_sae_pending_actions": int(df_site.get("n_sae_pending_actions", 0).sum()),
    }

    # prompt instructing the model to classify risk and suggest specific actions
    prompt = f"""
You are an operations co-pilot for a clinical trial.

Here is aggregated data for Study {study_id}, Site {site_id}:

- Subjects: {n_subj}
- Mean DQI (0–1): {mean_dqi:.3f}
- Percent Red-band subjects: {pct_red:.1f}%

Aggregate issue counts for this site:
{totals}

Using this information:

1. Briefly classify this site's risk level (Low/Medium/High) and why.
2. Suggest 3–6 specific next actions for the site CRA / data manager
   (e.g., "prioritize closure of outstanding safety queries", "focus monitoring on lab data", etc.).
3. Be concrete but concise; no more than 10 sentences total.
"""
    return _llm_call(prompt)


# ---------------------------------------------------------------------
# 3) Context-aware Q&A about the dashboard slice
# ---------------------------------------------------------------------

def chat_about_slice(
    user_question: str,
    df_slice: pd.DataFrame,
    level: str,
    study_id: str,
    site_id: str | None = None,
) -> str:
    """
    Answer a user's free-text question about the currently selected
    study/site slice of the data.

    df_slice: could be df_study (all sites) or df_site (one site).
    level: "study" or "site".
    """
    if df_slice.empty:
        return "The current selection has no subjects, so I cannot answer from the data."

    # Build a compact stats context for the LLM
    desc = df_slice[["dqi"]].describe().to_dict()  # basic DQI distribution stats
    band_counts = df_slice["dqi_band"].value_counts(normalize=False).to_dict()
    drivers_agg = (
        df_slice[
            [
                c
                for c in [
                    "n_missing_visits",
                    "n_missing_pages",
                    "n_open_queries",
                    "n_lab_issues",
                    "n_uncoded_terms",
                    "n_open_edrr_issues",
                    "n_sae_pending_actions",
                ]
                if c in df_slice.columns
            ]
        ]
        .sum()
        .to_dict()
    )

    # structured context object sent to the LLM
    context = {
        "level": level,
        "study_id": study_id,
        "site_id": site_id,
        "n_subjects": int(len(df_slice)),
        "dqi_stats": desc,
        "dqi_band_counts": band_counts,
        "issue_totals": drivers_agg,
    }

    # instruction prompt: answer using only provided context + generic knowledge
    prompt = f"""
You are an AI assistant helping a clinical data lead interpret a quality dashboard.

Here is JSON-style summary context for the current selection:
{context}

The user asks:
\"\"\"{user_question}\"\"\".

Answer ONLY using the information implied by the context and generic clinical-operations knowledge.
- If you refer to numbers, keep them approximate and clearly labeled as estimates.
- Be concise (5–10 sentences or bullet points).
- If something cannot be inferred from the context, say so explicitly.
"""
    return _llm_call(prompt)
