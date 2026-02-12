# data_ingestion.py
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd
import argparse

# Root folder containing all "Study X_CPID_Input Files - Anonymization" folders
# ROOT_DIR = Path(
#     r"C:\Users\Aritri Baidya\Desktop\Novartis\New folder"  # <<< CHANGE THIS
# )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class StudyPaths:
    study_id: str
    folder: Path
    cpid: Optional[Path] = None
    visits: Optional[Path] = None
    missing_lab: Optional[Path] = None
    sae: Optional[Path] = None
    inactivated: Optional[Path] = None
    missing_pages: Optional[Path] = None
    edrr: Optional[Path] = None
    medra: Optional[Path] = None
    whodd: Optional[Path] = None


def _match_first(folder: Path, patterns: List[str]) -> Optional[Path]:
    """
    Return first file whose name contains ALL substrings in `patterns`
    (case-insensitive). If none found, return None.
    """
    for f in folder.glob("*.xls*"):  # iterate over all Excel-like files in folder
        name = f.name.lower()
        if all(p.lower() in name for p in patterns):  # require every pattern to be present
            return f
    return None


def discover_study_folders(root: Path) -> List[StudyPaths]:

    """
    Find all study folders and infer paths for each required file type
    using robust substring matching based on your naming patterns.
    """
    study_paths: List[StudyPaths] = []

    for folder in root.iterdir():  # scan immediate children of ROOT_DIR
        if not folder.is_dir():
            continue
        # Accept "Study 1_...", "STUDY 20_..."
        name_lower = folder.name.lower()
        if "study" not in name_lower or "cpid_input files" not in name_lower:
            continue

        # Extract simple study ID (first number run in folder name)
        parts = [p for p in folder.name.replace("_", " ").split() if p.isdigit()]
        study_id = parts[0] if parts else folder.name  # fallback: whole folder name

        sp = StudyPaths(study_id=study_id, folder=folder)

        # Match patterns – tuned to the examples you gave
        sp.cpid = _match_first(folder, ["cpid", "edc", "metrics"])
        sp.visits = _match_first(folder, ["visit", "projection"])
        sp.missing_lab = _match_first(folder, ["missing", "lab"])
        sp.sae = _match_first(folder, ["sae", "dashboard"])
        sp.inactivated = _match_first(folder, ["inactivated"])
        sp.missing_pages = _match_first(folder, ["missing", "page"])
        if sp.missing_pages is None:
            sp.missing_pages = _match_first(folder, ["global", "missing", "pages"])
        sp.edrr = _match_first(folder, ["compiled", "edrr"])
        sp.medra = _match_first(folder, ["meddra"])
        sp.whodd = _match_first(folder, ["whodd"]) or _match_first(folder, ["who", "drug"])

        study_paths.append(sp)

    return study_paths

# ---------------------------------------------------------------------------
# Readers (kept very simple; you can add column cleaning here)
# ---------------------------------------------------------------------------

def _read_excel(path: Path, **kwargs) -> pd.DataFrame:
    df = pd.read_excel(path, **kwargs)  # generic Excel reader wrapper
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )  # normalize column names for easier downstream joins
    return df


def read_cpid(path: Path) -> pd.DataFrame:
    return _read_excel(path)


def read_visits(path: Path) -> pd.DataFrame:
    return _read_excel(path)


def read_missing_lab(path: Path) -> pd.DataFrame:
    return _read_excel(path)


def read_inactivated(path: Path) -> pd.DataFrame:
    return _read_excel(path)


def read_missing_pages(path: Path) -> pd.DataFrame:
    return _read_excel(path)


def read_edrr(path: Path) -> pd.DataFrame:
    return _read_excel(path)


def read_medra(path: Path) -> pd.DataFrame:
    return _read_excel(path)


def read_whodd(path: Path) -> pd.DataFrame:
    return _read_excel(path)


def read_sae(path: Path) -> Dict[str, pd.DataFrame]:
    """
    SAE layouts differ slightly by study. We treat every sheet that looks
    like DM vs Safety and label them heuristically.
    """
    xls = pd.ExcelFile(path)
    dm_sheet = None
    safety_sheet = None
    for s in xls.sheet_names:  # inspect all sheet names
        s_low = s.lower()
        if "dm" in s_low and dm_sheet is None:
            dm_sheet = s
        elif "safety" in s_low and safety_sheet is None:
            safety_sheet = s
    # Fallbacks: first and second sheet
    if dm_sheet is None and xls.sheet_names:
        dm_sheet = xls.sheet_names[0]
    if safety_sheet is None and len(xls.sheet_names) > 1:
        safety_sheet = xls.sheet_names[1]

    dm_df = _read_excel(path, sheet_name=dm_sheet)
    safety_df = _read_excel(path, sheet_name=safety_sheet) if safety_sheet else pd.DataFrame()
    return {"dm": dm_df, "safety": safety_df}

# ---------------------------------------------------------------------------
# Top-level loader: one study or all studies
# ---------------------------------------------------------------------------

def load_raw_for_study(sp: StudyPaths) -> Dict[str, pd.DataFrame]:
    """
    Load all source tables for one study into a dict of DataFrames.
    Keys: cpid, visits, missing_lab, sae_dm, sae_safety, inactivated,
          missing_pages, edrr, medra, whodd
    """
    out: Dict[str, pd.DataFrame] = {}

    if sp.cpid:
        out["cpid"] = read_cpid(sp.cpid)
        out["cpid"]["study_id"] = sp.study_id  # tag rows with study identifier

    if sp.visits:
        out["visits"] = read_visits(sp.visits)
        out["visits"]["study_id"] = sp.study_id

    if sp.missing_lab:
        out["missing_lab"] = read_missing_lab(sp.missing_lab)
        out["missing_lab"]["study_id"] = sp.study_id

    if sp.sae:
        sae_dict = read_sae(sp.sae)
        out["sae_dm"] = sae_dict["dm"]
        out["sae_dm"]["study_id"] = sp.study_id
        if not sae_dict["safety"].empty:
            out["sae_safety"] = sae_dict["safety"]
            out["sae_safety"]["study_id"] = sp.study_id

    if sp.inactivated:
        out["inactivated"] = read_inactivated(sp.inactivated)
        out["inactivated"]["study_id"] = sp.study_id

    if sp.missing_pages:
        out["missing_pages"] = read_missing_pages(sp.missing_pages)
        out["missing_pages"]["study_id"] = sp.study_id

    if sp.edrr:
        out["edrr"] = read_edrr(sp.edrr)
        out["edrr"]["study_id"] = sp.study_id

    if sp.medra:
        out["medra"] = read_medra(sp.medra)
        out["medra"]["study_id"] = sp.study_id

    if sp.whodd:
        out["whodd"] = read_whodd(sp.whodd)
        out["whodd"]["study_id"] = sp.study_id

    return out

def load_all_raw(root: Path) -> Dict[str, pd.DataFrame]:

    """
    Load and **concatenate** all studies into combined tables per source.
    """
    study_paths = discover_study_folders(root) # discover all study folders
    combined: Dict[str, List[pd.DataFrame]] = {}

    for sp in study_paths:
        raw_study = load_raw_for_study(sp)  # load each study individually
        for k, df in raw_study.items():
            combined.setdefault(k, []).append(df)  # collect per source type

    # concatenate per key
    out: Dict[str, pd.DataFrame] = {}
    for k, dfs in combined.items():
        out[k] = pd.concat([df for df in dfs if not df.empty], ignore_index=True)  # stack all studies for that source
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load study data")
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Path to the root folder containing study folders"
    )

    args = parser.parse_args()

    root_path = Path(args.root_dir)
    data = load_all_raw(root_path)

    print("Loaded datasets:")
    for k, v in data.items():
        print(f"{k}: {v.shape}")