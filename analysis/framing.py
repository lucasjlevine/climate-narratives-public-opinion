"""Aggregate framing / emotion prevalence for headlines."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.lexicons import FRAME_NAMES, binary_frame_presence, score_titles_binary


def attach_frame_scores(corpus: pd.DataFrame, *, use_alt: bool = False, title_col: str = "title_original") -> pd.DataFrame:
    scores = score_titles_binary(corpus[title_col].tolist(), use_alt=use_alt)
    scored = pd.DataFrame(scores)
    prefix = "alt_" if use_alt else ""
    scored = scored.rename(columns={c: f"{prefix}frame_{c}" for c in scored.columns})
    out = pd.concat([corpus.reset_index(drop=True), scored], axis=1)
    frame_cols = [f"{prefix}frame_{n}" for n in FRAME_NAMES]
    out[f"{prefix}n_frames"] = out[frame_cols].sum(axis=1)
    # Multi-label overlap indicator
    out[f"{prefix}multi_frame"] = (out[f"{prefix}n_frames"] >= 2).astype(int)
    return out


def prevalence_by(
    scored: pd.DataFrame,
    group_cols: str | list[str],
    frame_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Percentage of headlines with each frame present within groups."""
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    if frame_cols is None:
        frame_cols = [f"frame_{n}" for n in FRAME_NAMES]
    g = scored.groupby(group_cols, as_index=False)
    agg = g.agg(n_headlines=("id", "size"), **{c: (c, "mean") for c in frame_cols})
    # means of 0/1 columns are already proportions; express as percent
    for c in frame_cols:
        agg[c] = 100.0 * agg[c]
    return agg


def overlap_summary(scored: pd.DataFrame, frame_cols: list[str] | None = None) -> pd.DataFrame:
    if frame_cols is None:
        frame_cols = [f"frame_{n}" for n in FRAME_NAMES]
    n = len(scored)
    rows = []
    for i, a in enumerate(frame_cols):
        for b in frame_cols[i + 1 :]:
            both = ((scored[a] == 1) & (scored[b] == 1)).sum()
            rows.append(
                {
                    "frame_a": a.replace("frame_", ""),
                    "frame_b": b.replace("frame_", ""),
                    "n_both": int(both),
                    "pct_of_headlines": 100.0 * both / n if n else np.nan,
                }
            )
    return pd.DataFrame(rows).sort_values("pct_of_headlines", ascending=False)


def outlet_year_heatmap_table(
    scored: pd.DataFrame,
    frame: str,
    outlets: list[str] | None = None,
    top_n: int = 20,
) -> pd.DataFrame:
    """Wide table: outlet × year prevalence (%) for one frame."""
    col = frame if frame.startswith("frame_") else f"frame_{frame}"
    df = scored
    if outlets is None:
        outlets = (
            df.groupby("outlet")["id"]
            .size()
            .sort_values(ascending=False)
            .head(top_n)
            .index.tolist()
        )
    sub = df[df["outlet"].isin(outlets)]
    tab = (
        sub.groupby(["outlet", "year"])[col]
        .mean()
        .mul(100.0)
        .unstack("year")
        .reindex(outlets)
    )
    return tab
