
# Clinical Trial Data Quality & Operations Intelligence Platform

## Overview
This project provides an integrated analytics and AI‑assisted dashboard for monitoring clinical trial data quality and operational risks across multiple studies.  
It ingests heterogeneous clinical datasets, harmonizes them into a unified subject‑level dataset, computes a Data Quality Index (DQI), and visualizes insights through an interactive Streamlit dashboard with Generative AI support.

---

## Project Structure

```
project_root/
│
├── data_ingestion.py        # Raw Excel discovery & loading
├── feature_engineering.py   # Subject‑level aggregation & feature creation
├── scoring.py               # Clean patient logic & DQI computation
├── dashboard_app.py         # Streamlit dashboard
├── ai_utils.py              # LLM‑based summaries & recommendations
├── modeling_and_evaluation.ipynb (optional)
│
├── data/
│   ├── raw/                 # Study folders (Excel sources)
│   └── processed/           # Generated parquet snapshot
│
├── requirements.txt
└── README.md
```

---

## Prerequisites

### 1. Python Version
- Python **3.9 or higher** recommended.

### 2. Install Dependencies
Create a virtual environment (optional but recommended):

**Windows**
```
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux**
```
python3 -m venv venv
source venv/bin/activate
```

Install packages:

```
pip install -r requirements.txt
```

Example packages needed:
```
pandas
numpy
streamlit
plotly
langchain
langchain-huggingface
python-dotenv
openpyxl
pyarrow
```

---

## Data Setup

### Folder Structure
Place all study folders inside a single root directory:

```
data/raw/
├── Study 1_CPID_Input Files - Anonymization/
├── Study 2_CPID_Input Files - Anonymization/
└── Study 3_CPID_Input Files - Anonymization/
```

Each study folder should contain Excel files such as:
- CPID Metrics
- Visit Projection
- Missing Pages
- SAE Dashboard
- MedDRA / WHODD Coding
- EDRR Reports

---

## Running Data Ingestion (Optional Test)

To verify raw data loading:

```
python data_ingestion.py --root_dir "path/to/data/raw"
```

This will print dataset shapes and confirm successful ingestion.

---

## Running Data Ingestion (Optional Test)

After running data ingestion, run the .ipynb file:

```
Provide- "path/to/data/raw"
```

This will produce data/processed/subject_site_snapshot.parquet.

---


## Running the Dashboard

The dashboard is the primary output interface.

### Command
```
streamlit run dashboard_app.py -- --raw_root_dir "path/to/data/raw" --processed_path "data/processed/subject_site_snapshot.parquet"
```

### Explanation
- `--raw_root_dir` → Folder containing all study folders.
- `--processed_path` → Location where the aggregated parquet snapshot will be saved/read.

The first run may take longer because features and DQI are computed. Subsequent runs use cached parquet data.

---

## Dashboard Features

- Study‑level KPIs (Mean DQI, Clean Patient %)
- Site‑level Risk Analysis
- Subject‑level Drill‑down
- Interactive Filtering
- AI Study Summaries
- AI Site Recommendations
- Context‑aware Q&A

---

## AI Configuration (Optional)

Create a `.env` file in the project root if required by the HuggingFace endpoint:

```
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

---

## Rebuilding Snapshot

Inside the dashboard sidebar, enable:

```
Rebuild snapshot from raw files
```

This forces recalculation of features and DQI.

---

## Output Artifacts

- `subject_site_snapshot.parquet` → Unified dataset
- Interactive Streamlit dashboard
- AI‑generated narratives and recommendations

---

## Troubleshooting

**Issue:** Excel files not detected  
**Fix:** Ensure filenames contain keywords like `cpid`, `visit`, `missing`, `sae`, etc.

**Issue:** Dashboard slow on first run  
**Fix:** Normal behavior — snapshot building occurs once.

**Issue:** AI not responding  
**Fix:** Check internet connection and API token.

---

## Future Enhancements
- Real‑time streaming ingestion
- Predictive ML models
- Role‑based access control
- Automated alerts

---

## Conclusion
This platform enables proactive clinical trial data quality monitoring, reduces manual review effort, and augments operational decision‑making through analytics and AI‑assisted insights.
