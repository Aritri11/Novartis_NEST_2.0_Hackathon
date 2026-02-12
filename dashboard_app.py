# dashboard_app.py
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path
import argparse
import sys


from data_ingestion import load_all_raw
from feature_engineering import build_subject_snapshot
from scoring import compute_clean_patient_flags, compute_dqi
from ai_utils import summarize_study, recommend_actions_for_site, chat_about_slice


# ----------------------------------------------------------------------
# Data loading with caching
# ----------------------------------------------------------------------

@st.cache_data
def load_subject_snapshot(processed_path: Path, raw_root: Path, rebuild: bool = False) -> pd.DataFrame:
    """
    Load subject-level snapshot (all studies) with DQI.
    If rebuild=True or file missing, recompute from raw Excel files.
    """

    if processed_path.exists() and not rebuild:
        df = pd.read_parquet(processed_path)
        return df

    # recompute from raw source Excel files
    raw_all = load_all_raw(raw_root)
    df = build_subject_snapshot(raw_all)
    df = compute_clean_patient_flags(df)
    df = compute_dqi(df)

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed_path, index=False)
    return df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed_path",
        type=str,
        default="data/processed/subject_site_snapshot.parquet",
        help="Path to processed parquet snapshot",
    )
    parser.add_argument(
        "--raw_root_dir",
        type=str,
        default="data/raw",
        help="Root directory containing study folders",
    )

    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


# ----------------------------------------------------------------------
# Streamlit app
# ----------------------------------------------------------------------

def main():
    args = parse_args()

    processed_path = Path(args.processed_path)  # Uses CLI arg or default
    raw_root = Path(args.raw_root_dir)  # Uses CLI arg or default

    st.set_page_config(
        page_title="Clinical Trial Data Quality Dashboard",
        layout="wide",
    )

    st.title("Clinical Trial Data Quality Dashboard")
    st.markdown(
        "Multi-study view of **Data Quality Index (DQI)**, clean-patient status, "
        "and operational risk signals across sites and subjects."
    )

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.header("Controls")

        rebuild = st.checkbox("Rebuild snapshot from raw files", value=False)
        df = load_subject_snapshot(
            processed_path=processed_path,
            raw_root=raw_root,
            rebuild=rebuild
        )

        # keep only rows where study_id is purely numeric (drop labels / NaN)
        df["study_id_str"] = df["study_id"].astype(str)
        df = df[df["study_id_str"].str.fullmatch(r"\d+")]
        df["study_id"] = df["study_id_str"]
        df = df.drop(columns=["study_id_str"])

        # Use ALL numeric study IDs from the snapshot
        study_ids = sorted(df["study_id"].unique(), key=lambda x: int(x))
        study_labels = [f"Study {sid}" for sid in study_ids]
        label_to_id = dict(zip(study_labels, study_ids))

        selected_label = st.selectbox("Project / Study", study_labels)
        study = label_to_id[selected_label]

        # subset full snapshot to the selected study
        df_study = df[df["study_id"] == study]

        # site selector within the chosen study
        site_list = sorted(df_study["site_id"].astype(str).unique())
        site = st.selectbox("Site", ["All sites"] + site_list)

        band_filter = st.multiselect(
            "DQI band filter",
            options=["Green", "Amber", "Red"],
            default=["Green", "Amber", "Red"],
        )

    # Apply band filter
    if band_filter:
        df_study = df_study[df_study["dqi_band"].isin(band_filter)]

    # ---------------- Top-level KPIs ----------------
    st.markdown("### Study-level Overview")

    n_subjects = len(df_study)
    mean_dqi = df_study["dqi"].mean() if n_subjects else 0
    pct_clean = df_study["clean_patient"].mean() * 100 if n_subjects else 0
    pct_red = (df_study["dqi_band"] == "Red").mean() * 100 if n_subjects else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Study", str(study))
    col2.metric("Subjects", f"{n_subjects}")
    col3.metric("Mean DQI", f"{mean_dqi:0.3f}")
    col4.metric("% Clean Patients", f"{pct_clean:0.1f}%")

    st.caption(f"Red-band subjects: {pct_red:0.1f}% of all subjects in Study {study}.")

    st.markdown("---")

    # ---------------- Site-level view ----------------
    st.markdown("### Site-level Data Quality")

    if n_subjects == 0:
        # guard-rail when filters drop all subjects
        st.info("No subjects found for the selected filters.")
        return

    # aggregate subject-level metrics to site-level
    site_df = (
        df_study.groupby(["study_id", "site_id"], as_index=False)
        .agg(
            mean_dqi=("dqi", "mean"),
            pct_clean=("clean_patient", "mean"),
            n_subjects=("subject_id", "nunique"),
            n_red=("dqi_band", lambda x: (x == "Red").sum()),
        )
        .sort_values("mean_dqi")
    )

    # Bar chart of mean DQI by site
    fig_sites = px.bar(
        site_df,
        x="site_id",
        y="mean_dqi",
        color="mean_dqi",
        color_continuous_scale=["red", "orange", "yellow", "green"],
        title=f"Mean DQI by Site – Study {study}",
    )
    fig_sites.update_layout(
        xaxis_title="Site ID",
        yaxis_title="Mean DQI (0–1)",
        coloraxis_colorbar_title="DQI",
    )
    st.plotly_chart(fig_sites, use_container_width=True)

    # Site table
    st.write("Site summary table")
    site_display = site_df.assign(
        pct_clean=lambda d: (d["pct_clean"] * 100).round(1),
    )[["site_id", "n_subjects", "mean_dqi", "pct_clean", "n_red"]]
    st.dataframe(site_display, use_container_width=True)

    st.markdown("---")

    # ---------------- Subject-level drill-down ----------------
    st.markdown("### Subject-level Drill-down")

    if site != "All sites":
        df_site = df_study[df_study["site_id"].astype(str) == str(site)].copy()
        st.subheader(f"Site {site} – Subject-level Metrics")
    else:
        df_site = df_study.copy()
        st.subheader("All Sites – Subject-level Metrics")

    # DQI distribution histogram
    fig_hist = px.histogram(
        df_site,
        x="dqi",
        nbins=30,
        color="dqi_band",
        category_orders={"dqi_band": ["Red", "Amber", "Green"]},
        color_discrete_map={"Green": "green", "Amber": "orange", "Red": "red"},
        title="Subject-level DQI Distribution",
    )
    fig_hist.update_layout(xaxis_title="DQI", yaxis_title="Number of subjects")
    st.plotly_chart(fig_hist, use_container_width=True)

    # Subject table with key drivers
    cols = [
        "study_id",
        "site_id",
        "subject_id",
        "dqi",
        "dqi_band",
        "clean_patient",
        "n_missing_visits",
        "n_missing_pages",
        "n_open_queries",
        "n_lab_issues",
        "n_uncoded_terms",
        "n_open_edrr_issues",
        "n_sae_pending_actions",
    ]
    cols_present = [c for c in cols if c in df_site.columns]

    st.write("Subject detail table (sorted by lowest DQI first)")
    subj_display = df_site[cols_present].sort_values("dqi")
    st.dataframe(subj_display, use_container_width=True, height=500)

    # ---------------- AI Co-pilot (Generative & Agentic) ----------------
    st.markdown("### AI Co‑pilot")

    tab1, tab2, tab3 = st.tabs(
        ["Study summary", "Site recommendations", "Ask a question"]
    )

    # 1) Study-level narrative using LangChain
    with tab1:
        if st.button("Generate study-level summary", key="btn_study_summary"):
            with st.spinner("Generating narrative summary..."):
                summary_text = summarize_study(df_study, str(study))
            st.markdown(summary_text)

    # 2) Site-level operational recommendations
    with tab2:
        if site == "All sites":
            st.info("Select a specific site in the sidebar to get site-level recommendations.")
        else:
            if st.button(
                    f"Generate recommendations for Site {site}",
                    key="btn_site_reco",
            ):
                with st.spinner("Generating site recommendations..."):
                    reco_text = recommend_actions_for_site(df_site, str(study), str(site))
                st.markdown(reco_text)

    # 3) Context-aware Q&A about the current slice
    with tab3:
        user_q = st.text_area(
            "Ask a question about this study/site:",
            placeholder="Examples: 'Which issues are driving poor quality?' or "
                        "'What should we do first for this site?'",
            key="qa_input",
        )
        if st.button("Ask AI", key="btn_qa"):
            if not user_q.strip():
                st.warning("Please enter a question.")
            else:
                # choose appropriate slice based on site selection
                slice_df = df_site if site != "All sites" else df_study
                level = "site" if site != "All sites" else "study"
                with st.spinner("Thinking..."):
                    answer = chat_about_slice(
                        user_question=user_q,
                        df_slice=slice_df,
                        level=level,
                        study_id=str(study),
                        site_id=str(site) if site != "All sites" else None,
                    )
                st.markdown(answer)





if __name__ == "__main__":
    main()
