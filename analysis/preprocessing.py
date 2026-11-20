"""Load, standardize, and panel-construct MediaCloud headline exports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.lexicons import normalize_title, tokenize
from analysis.paths import (
    MEDIACLOUD_DIR,
    MEDIACLOUD_QUERY,
    STUDY_YEAR_MAX,
    STUDY_YEAR_MIN,
)


REQUIRED_COLS = ("id", "title", "publish_date", "media_name", "media_url", "url", "language")


@dataclass
class CorpusReport:
    n_raw_rows: int
    n_after_id_dedupe: int
    n_empty_or_invalid_title: int
    n_missing_date: int
    n_final: int
    n_outlets: int
    duplicate_id_rate: float
    year_min: int
    year_max: int


def discover_content_csvs(mc_dir: Path = MEDIACLOUD_DIR) -> list[Path]:
    return sorted(
        p
        for p in mc_dir.rglob("*-content.csv")
        if "scraped" not in p.parts and "scored" not in p.parts
    )


def load_raw_exports(
    mc_dir: Path = MEDIACLOUD_DIR,
    *,
    year_min: int = STUDY_YEAR_MIN,
    year_max: int = STUDY_YEAR_MAX,
) -> pd.DataFrame:
    """Concatenate yearly MediaCloud content CSVs; keep source file metadata."""
    csvs = discover_content_csvs(mc_dir)
    if not csvs:
        raise FileNotFoundError(f"No *-content.csv under {mc_dir}")

    frames: list[pd.DataFrame] = []
    for path in csvs:
        usecols = [c for c in REQUIRED_COLS if True]  # all required exist in exports
        df = pd.read_csv(path, usecols=list(REQUIRED_COLS), dtype=str, low_memory=False)
        df["source_file"] = str(path.relative_to(mc_dir))
        df["export_period"] = path.parent.name
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)
    raw["publish_date"] = pd.to_datetime(raw["publish_date"], errors="coerce", utc=False)
    # Filter to study window after parsing dates (exports can spill)
    if year_min is not None:
        raw = raw[(raw["publish_date"].isna()) | (raw["publish_date"].dt.year >= year_min)]
    if year_max is not None:
        raw = raw[(raw["publish_date"].isna()) | (raw["publish_date"].dt.year <= year_max)]
    return raw.reset_index(drop=True)


def clean_headline_corpus(raw: pd.DataFrame) -> tuple[pd.DataFrame, CorpusReport, dict]:
    """Standardize titles/dates, dedupe by story id, drop invalid titles.

    Preserves original title alongside normalized text. Does not drop rows
    merely because a URL is inaccessible — we analyze titles only.
    """
    n_raw = len(raw)
    work = raw.copy()

    work["title_original"] = work["title"].fillna("").astype(str)
    # Strip whitespace; keep original separately
    work["title_original"] = work["title_original"].str.strip()
    work["title_normalized"] = work["title_original"].map(normalize_title)

    # Invalid titles: empty / whitespace-only / placeholder-like
    invalid_mask = work["title_normalized"].str.len() < 3
    # Also flag titles that are only punctuation/digits after normalize
    only_nonalpha = ~work["title_normalized"].str.contains(r"[a-z]", regex=True, na=False)
    invalid_mask = invalid_mask | only_nonalpha
    n_invalid = int(invalid_mask.sum())

    n_missing_date = int(work["publish_date"].isna().sum())

    # Deduplicate by MediaCloud story id (preferred). Keep first occurrence.
    before_dedupe = len(work)
    n_dup_ids = int(work["id"].duplicated().sum())
    work = work.drop_duplicates(subset=["id"], keep="first")
    n_after_id = len(work)

    # Apply validity filters after dedupe so report is interpretable
    invalid_mask = work["title_normalized"].str.len() < 3
    only_nonalpha = ~work["title_normalized"].str.contains(r"[a-z]", regex=True, na=False)
    invalid_mask = invalid_mask | only_nonalpha
    work = work.loc[~invalid_mask].copy()
    work = work.dropna(subset=["publish_date"]).copy()

    work["year"] = work["publish_date"].dt.year.astype(int)
    work["month"] = work["publish_date"].dt.to_period("M").dt.to_timestamp()
    work["quarter"] = work["publish_date"].dt.to_period("Q").astype(str)
    work["quarter_start"] = work["publish_date"].dt.to_period("Q").dt.start_time
    work["n_tokens"] = work["title_original"].map(lambda t: len(tokenize(t)))
    work["media_name"] = work["media_name"].fillna("").astype(str).str.strip()
    work["outlet"] = work["media_name"]  # alias used throughout notebook

    report = CorpusReport(
        n_raw_rows=n_raw,
        n_after_id_dedupe=n_after_id,
        n_empty_or_invalid_title=n_invalid,
        n_missing_date=n_missing_date,
        n_final=len(work),
        n_outlets=int(work["outlet"].nunique()),
        duplicate_id_rate=float(n_dup_ids / before_dedupe) if before_dedupe else 0.0,
        year_min=int(work["year"].min()) if len(work) else np.nan,
        year_max=int(work["year"].max()) if len(work) else np.nan,
    )

    diagnostics = {
        "mediacloud_query": MEDIACLOUD_QUERY,
        "n_raw_rows": n_raw,
        "n_duplicate_ids_removed": n_dup_ids,
        "duplicate_id_rate": report.duplicate_id_rate,
        "n_empty_or_invalid_title": n_invalid,
        "n_missing_date": n_missing_date,
        "n_final": report.n_final,
        "n_outlets": report.n_outlets,
        "headline_token_mean": float(work["n_tokens"].mean()) if len(work) else np.nan,
        "headline_token_median": float(work["n_tokens"].median()) if len(work) else np.nan,
        "missing_outlet_rate": float((work["outlet"] == "").mean()) if len(work) else np.nan,
        "language_value_counts": work["language"].value_counts(dropna=False).head(10).to_dict()
        if "language" in work.columns
        else {},
    }
    return work.reset_index(drop=True), report, diagnostics


def syndication_candidates(corpus: pd.DataFrame) -> pd.DataFrame:
    """Flag same normalized title on the same day across different outlets.

    This is investigative, not used as a hard dedupe rule: wire copy and
    co-publication are substantively different from exact story-id duplicates.
    """
    g = (
        corpus.groupby(["title_normalized", "month"], as_index=False)
        .agg(
            n_rows=("id", "size"),
            n_outlets=("outlet", "nunique"),
            outlets=("outlet", lambda s: "|".join(sorted(set(s))[:8])),
        )
    )
    return g[(g["n_outlets"] >= 2) & (g["n_rows"] >= 2)].sort_values(
        "n_outlets", ascending=False
    )


def outlet_year_matrix(corpus: pd.DataFrame) -> pd.DataFrame:
    return (
        corpus.groupby(["year", "outlet"], as_index=False)
        .size()
        .rename(columns={"size": "n_articles"})
    )


def unique_outlets_by_year(corpus: pd.DataFrame) -> pd.DataFrame:
    return (
        corpus.groupby("year", as_index=False)
        .agg(n_articles=("id", "size"), n_outlets=("outlet", "nunique"))
        .sort_values("year")
    )


def fixed_outlet_panel(
    corpus: pd.DataFrame,
    *,
    year_min: int = STUDY_YEAR_MIN,
    year_max: int = STUDY_YEAR_MAX,
) -> tuple[set[str], pd.DataFrame]:
    """Outlets with ≥1 article in every calendar year of the study window."""
    years = list(range(year_min, year_max + 1))
    present: dict[int, set[str]] = {}
    for y, g in corpus.groupby("year"):
        if year_min <= int(y) <= year_max:
            present[int(y)] = set(g["outlet"].unique())
    missing_years = [y for y in years if y not in present]
    if missing_years:
        # Still compute intersection over available years, but warn via empty extras
        pass
    common = set.intersection(*(present[y] for y in years if y in present)) if present else set()
    panel = corpus[corpus["outlet"].isin(common) & corpus["year"].between(year_min, year_max)].copy()
    panel["in_fixed_panel"] = True
    return common, panel


def attach_panel_flag(corpus: pd.DataFrame, panel_outlets: set[str]) -> pd.DataFrame:
    out = corpus.copy()
    out["in_fixed_panel"] = out["outlet"].isin(panel_outlets)
    return out
