from __future__ import annotations

from pathlib import Path

# -------------------------- Main user settings --------------------------
# Change this one value to process a different element.
ELEMENT = "Ag"

# These can be relative to the repo or absolute paths on another disk, e.g.
# Path("/Volumes/ProjectData/geochemistry")
INPUT_DIR = Path("shp")
OUTPUT_DIR = Path("output")
# -----------------------------------------------------------------------


ELEMENT_FILE_STEM = ELEMENT.lower()
ELEMENT_INPUT_STEM = ELEMENT.upper()

INPUT_SHP = INPUT_DIR / f"{ELEMENT_INPUT_STEM}_Fusionn_imp.shp"
LOD_REFERENCE_SHP = INPUT_SHP

PROJECT_COLUMN = "NUMR_PROJ_"
RAW_VALUE_COLUMN = ELEMENT
IMPUTED_VALUE_COLUMN = f"{ELEMENT}_imp"
CENSORED_COLUMN = "is_censor"
LITHOLOGY_COLUMN = "CODE_TYPE_"

PHASE1_REGRESSION_PLOTS_DIRNAME = f"{ELEMENT_FILE_STEM}_phase1_regression_plots"
PHASE1_OUTPUT_NAME = f"{ELEMENT_FILE_STEM}_phase1_leveled_full_overlap.shp"
PHASE1_LOG_CSV_NAME = f"{ELEMENT_FILE_STEM}_phase1_leveling_log.csv"
PHASE1_SURVEY_QA_CSV_NAME = f"{ELEMENT_FILE_STEM}_phase1_survey_levelability.csv"
PHASE1_EXCLUDED_SURVEYS_CSV_NAME = f"{ELEMENT_FILE_STEM}_phase1_excluded_surveys.csv"

PHASE2_INPUT_SHP = OUTPUT_DIR / PHASE1_OUTPUT_NAME
PHASE2_REGRESSION_PLOTS_DIRNAME = f"{ELEMENT_FILE_STEM}_phase2_regression_plots"
PHASE2_QA_CSV_NAME = f"{ELEMENT_FILE_STEM}_phase2_survey_levelability.csv"
PHASE2_LOG_CSV_NAME = f"{ELEMENT_FILE_STEM}_phase2_leveling_log.csv"
PHASE2_EXCLUDED_CSV_NAME = f"{ELEMENT_FILE_STEM}_phase2_excluded_surveys.csv"
PHASE2_OUTPUT_NAME = f"{ELEMENT_FILE_STEM}_phase2_leveled_partial_overlap.shp"

PHASE3_INPUT_SHP = OUTPUT_DIR / PHASE2_OUTPUT_NAME
PHASE3_OUTPUT_TIF = OUTPUT_DIR / f"{ELEMENT_FILE_STEM}_phase3_imp_ordinary_kriging.tif"
PHASE3_VARIOGRAM_DIAGNOSTIC_DIR = OUTPUT_DIR / f"{ELEMENT_FILE_STEM}_phase3_variogram"
