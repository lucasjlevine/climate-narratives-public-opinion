#!/usr/bin/env python3
"""Build cleaned headline corpus, score frames/emotions, export annotation sample."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.annotation import export_labeling_spreadsheet, stratified_headline_sample
from analysis.emotion import PRIMARY_EMOTIONS, score_titles_emotions
from analysis.framing import attach_frame_scores
from analysis.lexicons import FRAME_NAMES
from analysis.paths import (
    ANNOTATION_SAMPLE_PATH,
    CORPUS_PATH,
    EMOTION_SCORES_PATH,
    FRAME_SCORES_PATH,
    OUTPUTS_DIR,
    RANDOM_SEED,
    STUDY_YEAR_MAX,
    STUDY_YEAR_MIN,
)
from analysis.preprocessing import (
    attach_panel_flag,
    clean_headline_corpus,
    fixed_outlet_panel,
    load_raw_exports,
    unique_outlets_by_year,
)


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--year-min", type=int, default=STUDY_YEAR_MIN)
    p.add_argument("--year-max", type=int, default=STUDY_YEAR_MAX)
    p.add_argument("--skip-emotions", action="store_true")
    p.add_argument("--sample-n", type=int, default=200)
    args = p.parse_args()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()

    log("Loading MediaCloud exports...")
    raw = load_raw_exports(year_min=args.year_min, year_max=args.year_max)
    log(f"  raw rows in window: {len(raw):,}")

    corpus, report, diagnostics = clean_headline_corpus(raw)
    panel_outlets, _panel = fixed_outlet_panel(corpus, year_min=args.year_min, year_max=args.year_max)
    corpus = attach_panel_flag(corpus, panel_outlets)
    corpus.to_parquet(CORPUS_PATH, index=False)
    log(f"Wrote {CORPUS_PATH} ({len(corpus):,} headlines; {report.n_outlets} outlets)")
    log(f"  Fixed panel outlets (all years {args.year_min}-{args.year_max}): {len(panel_outlets)}")

    coverage = unique_outlets_by_year(corpus)
    coverage.to_csv(OUTPUTS_DIR / "outlet_coverage_by_year.csv", index=False)
    pd.Series(sorted(panel_outlets), name="outlet").to_csv(
        OUTPUTS_DIR / "fixed_panel_outlets.csv", index=False
    )

    log("Scoring primary + alternative frame lexicons (binary presence)...")
    scored = attach_frame_scores(corpus, use_alt=False)
    # attach alt columns
    alt = attach_frame_scores(corpus[["id", "title_original"]].copy(), use_alt=True)
    alt_cols = [c for c in alt.columns if c.startswith("alt_")]
    scored = scored.merge(alt[["id"] + alt_cols], on="id", how="left")
    # Keep a lean score table for joins
    keep = [
        "id",
        "publish_date",
        "year",
        "month",
        "quarter",
        "quarter_start",
        "outlet",
        "in_fixed_panel",
        "n_tokens",
        "title_original",
    ]
    frame_cols = [f"frame_{n}" for n in FRAME_NAMES] + ["n_frames", "multi_frame"]
    frame_cols += [f"alt_frame_{n}" for n in FRAME_NAMES] + ["alt_n_frames", "alt_multi_frame"]
    scored[keep + frame_cols].to_parquet(FRAME_SCORES_PATH, index=False)
    log(f"Wrote {FRAME_SCORES_PATH}")

    if not args.skip_emotions:
        log("Scoring NRC emotion presence...")
        emos = score_titles_emotions(corpus["title_original"].tolist())
        emo_df = pd.DataFrame(emos)
        emo_df.insert(0, "id", corpus["id"].to_numpy())
        # merge calendar keys
        emo_out = corpus[keep].merge(emo_df, on="id", how="left")
        emo_out.to_parquet(EMOTION_SCORES_PATH, index=False)
        log(f"Wrote {EMOTION_SCORES_PATH}")
        # quick prevalence check
        for e in PRIMARY_EMOTIONS:
            log(f"  % headlines with NRC {e}: {100*emo_df[e].mean():.2f}")

    log(f"Drawing annotation sample (n={args.sample_n}, seed={RANDOM_SEED})...")
    sample, design, weights = stratified_headline_sample(
        corpus,
        n=args.sample_n,
        seed=RANDOM_SEED,
        year_min=args.year_min,
        year_max=args.year_max,
    )
    export_labeling_spreadsheet(sample, ANNOTATION_SAMPLE_PATH)
    weights.to_csv(OUTPUTS_DIR / "annotation_sample_design_weights.csv", index=False)
    design_dict = {
        "n_target": design.n_target,
        "n_drawn": design.n_drawn,
        "seed": design.seed,
        "n_years": design.n_years,
        "base_per_year": design.base_per_year,
        "remainder_years": design.remainder_years,
        "n_outlets_in_sample": design.n_outlets_in_sample,
        "year_allocation": {str(k): v for k, v in design.year_allocation.items()},
        "sampling": (
            "Year-stratified without replacement; within-year sampling probabilities "
            "proportional to 1/sqrt(outlet_count_in_year). Annotation fields exported empty."
        ),
    }
    (OUTPUTS_DIR / "annotation_sample_design.json").write_text(
        json.dumps(design_dict, indent=2), encoding="utf-8"
    )
    log(f"Wrote {ANNOTATION_SAMPLE_PATH} ({design.n_drawn} rows, {design.n_outlets_in_sample} outlets)")

    meta = {
        "diagnostics": {k: (v if not hasattr(v, "item") else v) for k, v in diagnostics.items()},
        "report": report.__dict__,
        "n_fixed_panel_outlets": len(panel_outlets),
        "elapsed_sec": round(time.monotonic() - t0, 1),
    }
    # language_value_counts may have non-serializable keys
    if "language_value_counts" in meta["diagnostics"]:
        meta["diagnostics"]["language_value_counts"] = {
            str(k): int(v) for k, v in meta["diagnostics"]["language_value_counts"].items()
        }
    (OUTPUTS_DIR / "corpus_build_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    log(f"Done in {time.monotonic() - t0:.1f}s")


if __name__ == "__main__":
    main()
