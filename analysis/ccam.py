"""CCAM survey preparation for RQ2.

Variable selection (from the released SPSS-label CSV)
-----------------------------------------------------
Concern
  - worry: "How worried are you about global warming?"
  - harm_US / harm_personally: perceived harm amount
  - happening / cause_recoded / sci_consensus: belief context (reported separately;
    not collapsed into a single "concern index")

Agency / efficacy
  CCAM has **no clean self-efficacy or collective-efficacy item**.
  We therefore do *not* invent an agency index. We report two related but
  distinct constructs separately:
  - discuss_GW: interpersonal discussion frequency (engagement / social agency proxy)
  - when_harm_US: perceived immediacy of harm (not efficacy; related to urgency)

Policy support (kept as separate outcomes; not averaged into one index)
  - reg_CO2_pollutant
  - fund_research
  - priority / priority_cleanenergy
  - transition_economy (high missingness; later waves only — report n carefully)

All wave aggregates use weight_wave. Media features are aligned to the
90 days preceding each wave's month start (wave labels like 'Apr-13').
"""

from __future__ import annotations

import re
from datetime import datetime

import numpy as np
import pandas as pd

from analysis.paths import CCAM_CSV

MONTHS = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}
WAVE_RE = re.compile(r"^([A-Za-z]{3})-(\d{2})$")
REFUSED = {"refused", "nan", ""}

# Exact item documentation for the notebook / paper
SURVEY_ITEMS = {
    "worry": {
        "construct": "public concern",
        "question": "How worried are you about global warming?",
        "values": ["Not at all worried", "Not very worried", "Somewhat worried", "Very worried"],
        "high_definition": "Somewhat or Very worried",
    },
    "harm_US": {
        "construct": "perceived collective harm (US)",
        "question": "How much do you think global warming will harm people in the United States?",
        "high_definition": "A moderate amount or A great deal",
    },
    "harm_personally": {
        "construct": "perceived personal harm",
        "question": "How much do you think global warming will harm you personally?",
        "high_definition": "A moderate amount or A great deal",
    },
    "when_harm_US": {
        "construct": "perceived immediacy of harm (NOT efficacy)",
        "question": "When do you think global warming will start to harm people in the United States?",
        "high_definition": "In 10 years or They are being harmed right now",
    },
    "discuss_GW": {
        "construct": "interpersonal discussion (engagement / weak agency proxy)",
        "question": "How often do you discuss global warming with your family and friends?",
        "high_definition": "Occasionally or Often",
        "caveat": "Not a direct efficacy measure; treat as communication engagement.",
    },
    "reg_CO2_pollutant": {
        "construct": "policy support — CO2 regulation",
        "question": "Do you support or oppose regulating carbon dioxide as a pollutant?",
        "high_definition": "Somewhat or Strongly support",
    },
    "fund_research": {
        "construct": "policy support — research funding",
        "question": "Do you support or oppose funding research into renewable energy sources?",
        "high_definition": "Somewhat or Strongly support",
    },
    "priority": {
        "construct": "priority of global warming for the president and Congress",
        "high_definition": "High or Very high",
    },
    "priority_cleanenergy": {
        "construct": "priority of developing clean energy",
        "high_definition": "High or Very high",
    },
    "transition_economy": {
        "construct": "support transitioning the economy from fossil fuels",
        "high_definition": "Somewhat or Strongly support",
        "caveat": "Missing for many early waves (~69% overall); do not pool with other policy items.",
    },
}


def wave_to_date(wave: str) -> pd.Timestamp | pd.NaT:
    if pd.isna(wave):
        return pd.NaT
    m = WAVE_RE.match(str(wave).strip())
    if not m:
        return pd.NaT
    mon, yy = m.group(1).title(), int(m.group(2))
    if mon not in MONTHS:
        return pd.NaT
    return pd.Timestamp(datetime(2000 + yy, MONTHS[mon], 1))


def _norm(x) -> str | None:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.lower() in REFUSED:
        return None
    return s


def _map_series(series: pd.Series, mapping: dict[str, float]) -> pd.Series:
    def one(v):
        s = _norm(v)
        if s is None:
            return np.nan
        if s in mapping:
            return mapping[s]
        key = s.lower()
        for k, val in mapping.items():
            if k.lower() == key:
                return val
        return np.nan

    return series.map(one)


WORRY_MAP = {
    "Not at all worried": 1,
    "Not very worried": 1,
    "Somewhat worried": 2,
    "Very worried": 2,
}
HARM_MAP = {
    "Don't know": 0,
    "Dont know": 0,
    "Not at all": 1,
    "Only a little": 1,
    "A moderate amount": 2,
    "A great deal": 2,
}
WHEN_HARM_MAP = {
    "Never": 1,
    "In 100 years": 1,
    "In 50 years": 1,
    "In 25 years": 1,
    "In 10 years": 2,
    "They are being harmed right now": 2,
}
OPPOSE_SUPPORT_MAP = {
    "Strongly oppose": 1,
    "Somewhat oppose": 1,
    "Somewhat support": 2,
    "Strongly support": 2,
}
PRIORITY_MAP = {"Low": 1, "Medium": 1, "High": 2, "Very high": 2}
DISCUSS_MAP = {"Never": 1, "Rarely": 1, "Occasionally": 2, "Often": 2}


BINARY_OUTCOMES = {
    "worry_high": ("worry_r", 2),
    "harm_personally_high": ("harm_personally_r", 2),
    "harm_US_high": ("harm_US_r", 2),
    "when_harm_soon": ("when_harm_US_r", 2),
    "discuss_often": ("discuss_GW_r", 2),
    "support_CO2_reg": ("reg_CO2_pollutant_r", 2),
    "support_fund_research": ("fund_research_r", 2),
    "support_transition": ("transition_economy_r", 2),
    "priority_high": ("priority_r", 2),
    "priority_cleanenergy_high": ("priority_cleanenergy_r", 2),
}


def load_ccam(path=CCAM_CSV) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def apply_recodes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["worry_r"] = _map_series(out["worry"], WORRY_MAP)
    for col in ("harm_personally", "harm_US", "harm_dev_countries", "harm_future_gen", "harm_plants_animals"):
        if col in out.columns:
            out[f"{col}_r"] = _map_series(out[col], HARM_MAP)
    out["when_harm_US_r"] = _map_series(out["when_harm_US"], WHEN_HARM_MAP)
    for col in ("reg_CO2_pollutant", "fund_research", "transition_economy", "reduce_tax", "generate_renewable"):
        if col in out.columns:
            out[f"{col}_r"] = _map_series(out[col], OPPOSE_SUPPORT_MAP)
    if "priority" in out.columns:
        out["priority_r"] = _map_series(out["priority"], PRIORITY_MAP)
    if "priority_cleanenergy" in out.columns:
        out["priority_cleanenergy_r"] = _map_series(out["priority_cleanenergy"], PRIORITY_MAP)
    if "discuss_GW" in out.columns:
        out["discuss_GW_r"] = _map_series(out["discuss_GW"], DISCUSS_MAP)
    out["wave_date"] = out["wave"].map(wave_to_date)
    return out


def weighted_prevalence(series: pd.Series, weights: pd.Series, high_value: float = 2) -> float:
    mask = series.notna() & weights.notna()
    if mask.sum() == 0:
        return np.nan
    s = series[mask]
    w = weights[mask]
    return float((w * (s == high_value).astype(float)).sum() / w.sum())


def wave_level_outcomes(ccam: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for wave, g in ccam.groupby("wave", sort=False):
        row = {
            "wave": wave,
            "wave_date": g["wave_date"].iloc[0],
            "year": g["year"].iloc[0] if "year" in g.columns else np.nan,
            "n": len(g),
            "n_weighted": float(g["weight_wave"].sum()) if "weight_wave" in g.columns else np.nan,
        }
        w = g["weight_wave"]
        for out_name, (col, high) in BINARY_OUTCOMES.items():
            if col not in g.columns:
                row[out_name] = np.nan
                row[f"n_{out_name}"] = 0
                continue
            valid = g[col].notna()
            row[f"n_{out_name}"] = int(valid.sum())
            row[out_name] = weighted_prevalence(g[col], w, high_value=high)
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .dropna(subset=["wave_date"])
        .sort_values("wave_date")
        .reset_index(drop=True)
    )


def media_window_features(
    media: pd.DataFrame,
    wave_dates: pd.Series,
    feature_cols: list[str],
    *,
    days: int = 90,
    date_col: str = "publish_date",
) -> pd.DataFrame:
    """Aggregate media features in the `days` before each wave_date (exclusive of wave month start+).

    Uses [wave_date - days, wave_date) so the survey field period is not mixed
    with post-field coverage. Missing archive months are left as NaN — never filled with 0.
    """
    m = media.dropna(subset=[date_col]).copy()
    m[date_col] = pd.to_datetime(m[date_col])
    rows = []
    for wd in wave_dates:
        if pd.isna(wd):
            rows.append({c: np.nan for c in feature_cols} | {"n_media": 0})
            continue
        start = wd - pd.Timedelta(days=days)
        window = m[(m[date_col] >= start) & (m[date_col] < wd)]
        row = {"n_media": int(len(window))}
        if len(window) == 0:
            row.update({c: np.nan for c in feature_cols})
        else:
            for c in feature_cols:
                # frame_* binary cols → prevalence in %; others → mean
                if c.startswith("frame_") or c.startswith("nrc_") or c.startswith("alt_frame_"):
                    row[c] = float(100.0 * window[c].mean())
                else:
                    row[c] = float(window[c].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def spearman_pair_table(df: pd.DataFrame, x_cols: list[str], y_cols: list[str]) -> pd.DataFrame:
    """Spearman correlations with n and p-values; no causal language."""
    from scipy import stats

    rows = []
    for x in x_cols:
        for y in y_cols:
            sub = df[[x, y]].dropna()
            if len(sub) < 5:
                rho, p = np.nan, np.nan
            else:
                rho, p = stats.spearmanr(sub[x], sub[y])
            rows.append({"media": x, "opinion": y, "n_waves": len(sub), "spearman_rho": rho, "p_value": p})
    return pd.DataFrame(rows)
