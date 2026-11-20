"""Project paths. Raw MediaCloud/CCAM inputs live under inputs/; outputs under project root."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUTS_DIR = PROJECT_ROOT / "inputs"
OLD_ROOT = PROJECT_ROOT / "OLD"

MEDIACLOUD_DIR = INPUTS_DIR / "mediacloud"
CCAM_CSV = INPUTS_DIR / "ccam" / "CCAM_SPSS_Data_2008-2025.csv"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DOCS_DIR = PROJECT_ROOT / "docs"

CORPUS_PATH = OUTPUTS_DIR / "titles_corpus.parquet"
FRAME_SCORES_PATH = OUTPUTS_DIR / "headline_frame_scores.parquet"
EMOTION_SCORES_PATH = OUTPUTS_DIR / "headline_emotion_scores.parquet"
ANNOTATION_SAMPLE_PATH = OUTPUTS_DIR / "annotation_sample_200.csv"
CODEBOOK_PATH = DOCS_DIR / "annotation_codebook.md"

RANDOM_SEED = 42
STUDY_YEAR_MIN = 2008
STUDY_YEAR_MAX = 2025

# Documented MediaCloud query (identical across yearly *-content.txt files)
MEDIACLOUD_QUERY = (
    '"climate change" OR "global warming" OR "climate disaster" OR "climate crisis"'
)
MEDIACLOUD_COLLECTION_ID = 34412234
