"""Robustness / sensitivity comparisons for RQ1–RQ2."""

from __future__ import annotations

import pandas as pd

from analysis.framing import prevalence_by
from analysis.lexicons import FRAME_NAMES


def compare_full_vs_panel(
    scored_full: pd.DataFrame,
    scored_panel: pd.DataFrame,
    *,
    time_col: str = "year",
) -> pd.DataFrame:
    """Side-by-side annual frame prevalence (%) for full corpus vs fixed panel."""
    cols = [f"frame_{n}" for n in FRAME_NAMES]
    full = prevalence_by(scored_full, time_col, cols).rename(
        columns={c: f"full_{c}" for c in cols} | {"n_headlines": "n_full"}
    )
    panel = prevalence_by(scored_panel, time_col, cols).rename(
        columns={c: f"panel_{c}" for c in cols} | {"n_headlines": "n_panel"}
    )
    out = full.merge(panel, on=time_col, how="outer")
    for n in FRAME_NAMES:
        out[f"delta_{n}"] = out[f"full_frame_{n}"] - out[f"panel_frame_{n}"]
    return out.sort_values(time_col)


def compare_primary_vs_alt(scored_with_both: pd.DataFrame, time_col: str = "year") -> pd.DataFrame:
    primary = [f"frame_{n}" for n in FRAME_NAMES]
    alt = [f"alt_frame_{n}" for n in FRAME_NAMES]
    p = prevalence_by(scored_with_both, time_col, primary)
    a = prevalence_by(scored_with_both, time_col, alt)
    a = a.rename(columns={c: c.replace("alt_", "alt_") for c in a.columns})
    out = p.merge(a, on=time_col, suffixes=("_primary", "_alt"))
    return out


def outlet_composition_shift(corpus: pd.DataFrame) -> pd.DataFrame:
    """Share of annual volume from outlets that are *new* relative to 2008 set."""
    base = set(corpus.loc[corpus["year"] == corpus["year"].min(), "outlet"].unique())
    rows = []
    for y, g in corpus.groupby("year"):
        outlets = set(g["outlet"].unique())
        new = outlets - base
        n_new = g["outlet"].isin(new).sum()
        rows.append(
            {
                "year": int(y),
                "n_articles": len(g),
                "n_outlets": len(outlets),
                "n_outlets_not_in_base_year": len(new),
                "pct_articles_from_non_base_outlets": 100.0 * n_new / len(g) if len(g) else float("nan"),
                "mean_tokens": float(g["n_tokens"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("year")


def headline_length_by_year(corpus: pd.DataFrame) -> pd.DataFrame:
    return (
        corpus.groupby("year", as_index=False)
        .agg(
            n=("id", "size"),
            mean_tokens=("n_tokens", "mean"),
            median_tokens=("n_tokens", "median"),
            p90_tokens=("n_tokens", lambda s: s.quantile(0.9)),
        )
        .sort_values("year")
    )
