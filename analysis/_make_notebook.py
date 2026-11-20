#!/usr/bin/env python3
"""Generate climate_narratives_public_opinion.ipynb without nbformat dependency."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "climate_narratives_public_opinion.ipynb"


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _split(source.strip() + "\n"),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _split(source.strip() + "\n"),
    }


def _split(text: str) -> list[str]:
    if not text.endswith("\n"):
        text += "\n"
    lines = text.splitlines(keepends=True)
    return lines


cells: list[dict] = []

cells.append(md("""
# How Climate Change Narratives Shape Public Opinion

**Revised research notebook (titles-only MediaCloud + CCAM, 2008–2025)**

This notebook answers two research questions with transparent, reproducible methods. Prior exploratory work is preserved under `OLD/`; methodological changes are documented in `docs/METHODOLOGY_CHANGES.md`.

> **Integrity note.** Results are computed from the actual corpora. Empty human-annotation fields are intentional. ClimateBERT and BERTopic are *not* treated as primary evidence for headline framing (see Sections 7 and methodology notes).
"""))

cells.append(md("""
## 1. Research questions and methodology

### Research questions
- **RQ1:** How have framing and emotional tone in American climate change news *headlines* evolved over time? Has coverage shifted from scientific uncertainty toward political/economic issues and actionable solutions?
- **RQ2:** Are changes in these narratives associated with changes in public concern, perceived personal/collective agency-related measures, and support for climate policies?

### Data
| Source | Role | Notes |
|---|---|---|
| MediaCloud US online news exports | Headlines 2008–2025 | Query: `"climate change" OR "global warming" OR "climate disaster" OR "climate crisis"`; collection `34412234` |
| Yale/George Mason CCAM | Public opinion waves 2008–2025 | Respondent file with `weight_wave` |

We analyze **article titles only**. We do not scrape or score article bodies in this notebook.

### Method map
| Method | RQ | Role |
|---|---|---|
| Custom 6-category frame lexicons (binary presence → % headlines) | RQ1 | Primary framing measure |
| Fixed outlet panel vs full corpus | RQ1/RQ2 | Archive-composition robustness |
| Human annotation (n=200) + codebook | RQ1 validation | Required; labels start empty |
| NRC emotion word presence | RQ1 | Exploratory tone; not reader emotion |
| ClimateBERT | — | Documented and **excluded as primary** |
| Topic modeling (BERTopic) | — | **Omitted** (headlines ≈ 10 tokens) |
| Wave-aligned media ↔ CCAM correlations | RQ2 | Associations only, not causation |

### Important limitations
1. MediaCloud’s outlet inventory and volume change over time — composition shifts can mimic narrative change.
2. Missing archive coverage ≠ zero real-world coverage.
3. Lexicon hits are cues, not proof of a headline’s primary frame.
4. CCAM has no clean efficacy item; agency-related analyses use carefully labeled proxies.
5. Headline-only text is thin; conclusions should stay appropriately modest.
6. MediaCloud matching can hit **article bodies**, so many exported titles do not themselves contain climate keywords. Headline framing measures therefore describe the titles of climate-*query* stories, not only titles that mention climate explicitly.
"""))

cells.append(code("""
from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "analysis").exists():
    for cand in [PROJECT_ROOT, *PROJECT_ROOT.parents]:
        if (cand / "analysis").exists() and (cand / "OLD").exists():
            PROJECT_ROOT = cand
            break

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.paths import (
    ANNOTATION_SAMPLE_PATH,
    CCAM_CSV,
    CORPUS_PATH,
    DOCS_DIR,
    EMOTION_SCORES_PATH,
    FRAME_SCORES_PATH,
    MEDIACLOUD_QUERY,
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
    outlet_year_matrix,
    syndication_candidates,
    unique_outlets_by_year,
)
from analysis.lexicons import AMBIGUOUS_TERMS_NOTES, FRAME_LEXICONS, FRAME_NAMES
from analysis.framing import (
    attach_frame_scores,
    outlet_year_heatmap_table,
    overlap_summary,
    prevalence_by,
)
from analysis.emotion import PRIMARY_EMOTIONS
from analysis.annotation import (
    ANNOTATION_FIELDS,
    PRIMARY_FRAME_LABELS,
    cohens_kappa,
    confusion_against_consensus,
    export_labeling_spreadsheet,
    stratified_headline_sample,
)
from analysis.ccam import (
    SURVEY_ITEMS,
    apply_recodes,
    load_ccam,
    media_window_features,
    spearman_pair_table,
    wave_level_outcomes,
)
from analysis.robustness import (
    compare_full_vs_panel,
    headline_length_by_year,
    outlet_composition_shift,
)
from analysis.climatebert_review import decision_summary

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.figsize"] = (11, 4.5)
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
print("PROJECT_ROOT =", PROJECT_ROOT)
print("Study window:", STUDY_YEAR_MIN, "-", STUDY_YEAR_MAX)
print("Random seed:", RANDOM_SEED)
print("MediaCloud query:", MEDIACLOUD_QUERY)
"""))

cells.append(md("""
## 2. Data loading and preprocessing

### Goals
- Combine yearly MediaCloud `*-content.csv` exports
- Standardize dates, titles, outlets, story IDs
- Preserve `title_original` alongside `title_normalized`
- Deduplicate on MediaCloud `id`
- Investigate cross-outlet syndication separately (not hard-dropped)
- Create year / month / quarter fields
- Report missingness, duplicate rates, lengths, outlet counts

If `outputs/titles_corpus.parquet` already exists from `python -m analysis.build_and_score`, we load it; otherwise we build from the CSVs.
"""))

cells.append(code("""
if CORPUS_PATH.exists():
    corpus = pd.read_parquet(CORPUS_PATH)
    print(f"Loaded cached corpus: {CORPUS_PATH} ({len(corpus):,} rows)")
else:
    print("Building corpus from MediaCloud CSVs...")
    raw = load_raw_exports(year_min=STUDY_YEAR_MIN, year_max=STUDY_YEAR_MAX)
    corpus, report, diagnostics = clean_headline_corpus(raw)
    panel_outlets, _ = fixed_outlet_panel(corpus)
    corpus = attach_panel_flag(corpus, panel_outlets)
    corpus.to_parquet(CORPUS_PATH, index=False)
    print(report)
    print(diagnostics)

if "in_fixed_panel" not in corpus.columns:
    panel_outlets, _ = fixed_outlet_panel(corpus)
    corpus = attach_panel_flag(corpus, panel_outlets)
else:
    panel_outlets = set(corpus.loc[corpus["in_fixed_panel"], "outlet"].unique())

print(corpus[["id", "publish_date", "outlet", "year", "n_tokens", "title_original"]].head(3))
print("Columns:", list(corpus.columns))
"""))

cells.append(code("""
n = len(corpus)
print("=== Corpus diagnostics (study window) ===")
print(f"Headlines: {n:,}")
print(f"Unique outlets: {corpus['outlet'].nunique():,}")
print(f"Fixed-panel outlets: {len(panel_outlets):,}")
print(
    f"Fixed-panel headlines: {corpus['in_fixed_panel'].sum():,} "
    f"({100 * corpus['in_fixed_panel'].mean():.1f}%)"
)
print(f"Date range: {corpus['publish_date'].min()} -> {corpus['publish_date'].max()}")
print(f"Missing outlet name: {(corpus['outlet'] == '').sum():,}")
print(f"Mean / median tokens: {corpus['n_tokens'].mean():.2f} / {corpus['n_tokens'].median():.1f}")
print(corpus["n_tokens"].describe())
print(f"Duplicate IDs remaining: {corpus['id'].duplicated().sum()}")
climate_kw = corpus["title_original"].str.contains(
    r"climate|global warming|greenhouse|climate crisis|climate disaster",
    case=False,
    na=False,
)
print(
    f"Titles containing climate/warming/greenhouse/crisis/disaster keywords: "
    f"{climate_kw.sum():,} ({100 * climate_kw.mean():.1f}%)"
)
print(
    "Note: MediaCloud query matching can include body hits, so <100% keyworded titles "
    "is expected — and is a core headline-only limitation."
)
coverage = unique_outlets_by_year(corpus)
coverage
"""))

cells.append(code("""
syn = syndication_candidates(corpus)
print(f"Title-month groups with >=2 outlets: {len(syn):,}")
print("Top syndication-like title groups (inspect; do not auto-drop):")
syn.head(500).to_csv(OUTPUTS_DIR / "syndication_candidates_head.csv", index=False)
syn.head(15)
"""))

cells.append(md("""
## 3. Exploratory data analysis

### Archive and outlet composition
MediaCloud source coverage expands and contracts. **Do not interpret a thin year as the absence of climate news in the real world** — it may reflect archive composition. We therefore:

1. Plot annual volume and unique outlets
2. Identify outlets present in every year 2008–2025
3. Carry a **fixed outlet panel** through later analyses
"""))

cells.append(code("""
fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
axes[0].bar(coverage["year"], coverage["n_articles"], color="#4C72B0", width=0.8)
axes[0].set_ylabel("Headlines")
axes[0].set_title("Annual climate-headline volume in MediaCloud export")
axes[1].plot(coverage["year"], coverage["n_outlets"], marker="o", color="#DD8452")
axes[1].axhline(
    len(panel_outlets), color="gray", ls="--", lw=1, label=f"Fixed panel (n={len(panel_outlets)})"
)
axes[1].set_ylabel("Unique outlets")
axes[1].set_xlabel("Year")
axes[1].set_title("Unique outlets per year (archive composition)")
axes[1].legend()
plt.tight_layout()
plt.show()
print("Fixed-panel outlets (first 25):")
print(sorted(panel_outlets)[:25])
"""))

cells.append(code("""
oy = outlet_year_matrix(corpus)
top_outlets = (
    corpus.groupby("outlet")["id"].size().sort_values(ascending=False).head(20).index.tolist()
)
pivot = (
    oy[oy["outlet"].isin(top_outlets)]
    .pivot(index="outlet", columns="year", values="n_articles")
    .reindex(top_outlets)
)
fig, ax = plt.subplots(figsize=(12, 7))
sns.heatmap(
    np.log10(1 + pivot.fillna(0)),
    ax=ax,
    cmap="Blues",
    cbar_kws={"label": "log10(1 + count)"},
)
ax.set_title("Top-20 outlets: log article counts by year")
ax.set_xlabel("Year")
ax.set_ylabel("Outlet")
plt.tight_layout()
plt.show()
"""))

cells.append(code("""
comp = outlet_composition_shift(corpus)
lengths = headline_length_by_year(corpus)
fig, ax1 = plt.subplots(figsize=(11, 4))
ax1.plot(comp["year"], comp["pct_articles_from_non_base_outlets"], marker="o", color="#4C72B0")
ax1.set_ylabel("% articles from outlets not in first-year set")
ax1.set_xlabel("Year")
ax1.set_title("Outlet-composition shift relative to first study year")
plt.tight_layout()
plt.show()
display(comp.merge(lengths.drop(columns=["n"]), on="year"))
print(
    "Interpretation: rising share from non-base outlets means temporal trends on the FULL "
    "corpus can partly reflect who is in the archive, not only how the same outlets write."
)
"""))

cells.append(md("""
## 4. Human annotation and codebook

### Why this comes before trusting automated frames
Dictionary methods are transparent but imperfect. Human labels define the constructs (primary frame, tone, risk/opportunity) against which automated scores are checked.

**Codebook:** [`docs/annotation_codebook.md`](docs/annotation_codebook.md) — written to distinguish scientific uncertainty from denial, and impacts from negative sentiment, *before* model-assisted adjudication.

### Sampling design
- N = 200 headlines
- Stratified by year across 2008–2025
- Within-year sampling weights proportional to `1/sqrt(outlet frequency)` for outlet diversity
- Seed = 42
- Annotation columns exported **empty** (no model predictions)
"""))

cells.append(code("""
print("Codebook:", DOCS_DIR / "annotation_codebook.md")
print("Primary frame label set:", PRIMARY_FRAME_LABELS)

if ANNOTATION_SAMPLE_PATH.exists():
    ann = pd.read_csv(ANNOTATION_SAMPLE_PATH)
    print(f"Loaded existing sample: {ANNOTATION_SAMPLE_PATH} ({len(ann)} rows)")
else:
    sample, design, weights = stratified_headline_sample(
        corpus, n=200, seed=RANDOM_SEED, year_min=STUDY_YEAR_MIN, year_max=STUDY_YEAR_MAX
    )
    ann = export_labeling_spreadsheet(sample, ANNOTATION_SAMPLE_PATH)
    weights.to_csv(OUTPUTS_DIR / "annotation_sample_design_weights.csv", index=False)
    print(design)
    print(f"Wrote {ANNOTATION_SAMPLE_PATH}")

print("Year allocation in sample:")
display(ann.groupby("year").size().rename("n").reset_index())
print(f"Outlets represented: {ann['outlet'].nunique()}")
filled = (
    ann[[c for c in ann.columns if c.startswith("annotator1_")]]
    .astype(str)
    .apply(lambda s: s.str.len() > 0)
    .any()
    .any()
)
print("Annotator1 fields pre-filled?", bool(filled))
ann.head(3)
"""))

cells.append(code("""
def annotation_agreement_report(path=ANNOTATION_SAMPLE_PATH):
    df = pd.read_csv(path)
    k = cohens_kappa(df["annotator1_primary_frame"], df["annotator2_primary_frame"])
    both = (
        df["annotator1_primary_frame"].astype(str).str.len().gt(0)
        & df["annotator2_primary_frame"].astype(str).str.len().gt(0)
    )
    print(f"Rows with both primary frames: {both.sum()} / {len(df)}")
    if pd.isna(k):
        print("Cohen's kappa: n/a (no paired labels yet)")
        print("Manual task remaining: independent dual annotation + adjudication.")
        return None
    print(f"Cohen's kappa (primary frame): {k:.3f}")
    ct = pd.crosstab(
        df.loc[both, "annotator1_primary_frame"],
        df.loc[both, "annotator2_primary_frame"],
    )
    display(ct)
    return k

annotation_agreement_report()
"""))

cells.append(md("""
## 5. Framing analysis (RQ1)

### Dictionaries
Six categories: **science**, **scientific_uncertainty**, **skepticism_denial**, **politics**, **economics**, **solutions**.

Full term lists and ambiguous-term notes live in `analysis/lexicons.py`. Critical coding distinctions:
- `debate` / bare `skeptical` are **not** treated as denial
- Bare `gas` / `oil` / `ira` / `evidence` were removed or restricted to reduce false positives
- Presence of a lexicon term is not proof of primary frame

### Metric
For each headline: binary indicator per category.  
Aggregate: **percentage of headlines** containing each category (annual and quarterly).  
We do **not** use keyword hits per 1,000 tokens.
"""))

cells.append(code("""
for name, terms in FRAME_LEXICONS.items():
    print(f"{name}: {len(terms)} terms")
print("\\nAmbiguous-term notes:")
for k, v in AMBIGUOUS_TERMS_NOTES.items():
    print(f"- {k}: {v}")
"""))

cells.append(code("""
if FRAME_SCORES_PATH.exists():
    frames = pd.read_parquet(FRAME_SCORES_PATH)
    print(f"Loaded frame scores: {FRAME_SCORES_PATH} ({len(frames):,})")
else:
    print("Scoring frames on full corpus...")
    frames = attach_frame_scores(corpus, use_alt=False)
    alt = attach_frame_scores(corpus[["id", "title_original"]], use_alt=True)
    alt_cols = [c for c in alt.columns if c.startswith("alt_")]
    frames = frames.merge(alt[["id"] + alt_cols], on="id", how="left")
    keep_cols = [
        "id", "publish_date", "year", "month", "quarter", "quarter_start",
        "outlet", "in_fixed_panel", "n_tokens", "title_original",
    ] + [f"frame_{n}" for n in FRAME_NAMES] + ["n_frames", "multi_frame"] + alt_cols
    frames[keep_cols].to_parquet(FRAME_SCORES_PATH, index=False)
    frames = pd.read_parquet(FRAME_SCORES_PATH)

frame_cols = [f"frame_{n}" for n in FRAME_NAMES]
print("Overall % headlines with each frame:")
print((100 * frames[frame_cols].mean()).round(2))
print(f"% headlines with >=1 frame hit: {100 * (frames['n_frames'] > 0).mean():.2f}")
print(f"% multi-frame headlines: {100 * frames['multi_frame'].mean():.2f}")
"""))

cells.append(code("""
annual_full = prevalence_by(frames, "year", frame_cols)
annual_panel = prevalence_by(frames[frames["in_fixed_panel"]], "year", frame_cols)
quarterly_full = prevalence_by(frames, "quarter_start", frame_cols)

fig, ax = plt.subplots(figsize=(11, 5))
for col in frame_cols:
    ax.plot(annual_full["year"], annual_full[col], marker="o", label=col.replace("frame_", ""))
ax.set_ylabel("% of headlines")
ax.set_xlabel("Year")
ax.set_title("RQ1: Annual framing prevalence (full corpus)")
ax.legend(ncols=2, fontsize=9)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(11, 5))
for col in frame_cols:
    ax.plot(annual_panel["year"], annual_panel[col], marker="o", label=col.replace("frame_", ""))
ax.set_ylabel("% of headlines")
ax.set_xlabel("Year")
ax.set_title("RQ1: Annual framing prevalence (fixed outlet panel)")
ax.legend(ncols=2, fontsize=9)
plt.tight_layout()
plt.show()

annual_full.to_csv(OUTPUTS_DIR / "framing_prevalence_annual_full.csv", index=False)
annual_panel.to_csv(OUTPUTS_DIR / "framing_prevalence_annual_panel.csv", index=False)
quarterly_full.to_csv(OUTPUTS_DIR / "framing_prevalence_quarterly_full.csv", index=False)
"""))

cells.append(code("""
focus = ["frame_politics", "frame_economics", "frame_solutions", "frame_scientific_uncertainty"]
fig, ax = plt.subplots(figsize=(12, 4.5))
for col in focus:
    ax.plot(
        quarterly_full["quarter_start"],
        quarterly_full[col],
        lw=1.2,
        label=col.replace("frame_", ""),
    )
ax.set_ylabel("% of headlines")
ax.set_title("Quarterly framing prevalence (selected categories, full corpus)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()
"""))

cells.append(code("""
ov = overlap_summary(frames, frame_cols)
print("Frame co-occurrence (% of all headlines):")
display(ov.head(15))

heat = outlet_year_heatmap_table(frames, "politics", top_n=15)
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(
    heat,
    ax=ax,
    cmap="YlOrRd",
    vmin=0,
    vmax=max(40, float(np.nanpercentile(heat.values, 95))),
)
ax.set_title("Politics-frame prevalence (%) by outlet and year (top-15 outlets)")
plt.tight_layout()
plt.show()
"""))

cells.append(code("""
for cat in FRAME_NAMES:
    hits = frames[frames[f"frame_{cat}"] == 1][["year", "outlet", "title_original"]].head(3)
    print(f"=== {cat} examples ===")
    display(hits)
"""))

cells.append(md("""
## 6. Emotional language analysis (RQ1, exploratory)

We use the **NRC Word-Emotion Association Lexicon** to mark whether a headline contains at least one word associated with each emotion, then plot the **percentage of headlines** with each emotion over time.

### Interpretation limits
- Lexical association is not actual reader emotion
- Disaster words inflate fear/sadness without meaning the headline negatively evaluates climate change as a political object
- Validate against human `emotional_language` / `tone` fields when available
- **VADER is not used** as a primary method here
"""))

cells.append(code("""
if EMOTION_SCORES_PATH.exists():
    emos = pd.read_parquet(EMOTION_SCORES_PATH)
    print(f"Loaded emotion scores: {len(emos):,}")
else:
    from analysis.emotion import score_titles_emotions

    print("Scoring NRC emotion presence (may take several minutes)...")
    rows = score_titles_emotions(corpus["title_original"].tolist())
    emos = pd.DataFrame(rows)
    emos.insert(0, "id", corpus["id"].to_numpy())
    emos = corpus[
        [
            "id",
            "publish_date",
            "year",
            "month",
            "quarter_start",
            "outlet",
            "in_fixed_panel",
            "n_tokens",
            "title_original",
        ]
    ].merge(emos, on="id")
    emos.to_parquet(EMOTION_SCORES_PATH, index=False)

emo_cols = list(PRIMARY_EMOTIONS)
annual_emo = emos.groupby("year", as_index=False).agg(
    n=("id", "size"), **{e: (e, "mean") for e in emo_cols}
)
for e in emo_cols:
    annual_emo[e] = 100 * annual_emo[e]

fig, ax = plt.subplots(figsize=(11, 5))
for e in emo_cols:
    ax.plot(annual_emo["year"], annual_emo[e], marker="o", label=e)
ax.set_ylabel("% of headlines with >=1 NRC word")
ax.set_xlabel("Year")
ax.set_title("Exploratory NRC emotion presence over time (full corpus)")
ax.legend(ncols=3, fontsize=9)
plt.tight_layout()
plt.show()
annual_emo.to_csv(OUTPUTS_DIR / "nrc_prevalence_annual.csv", index=False)
"""))

cells.append(md("""
## 7. ClimateBERT — suitability review (excluded as primary)

Before using ClimateBERT we inspected the checkpoint, training data, labels, and intended task.
"""))

cells.append(code("""
summary = decision_summary()
for k, v in summary.items():
    print(f"{k}: {v}")

print()
print("DECISION: Do not use ClimateBERT sentiment as a primary RQ1 risk/opportunity series")
print("for headlines.")
print()
print("Optional later step (after human adjudication): run the sentiment head on the 200")
print("labeled titles only, then report precision / recall / F1 and a confusion matrix")
print("against adjudicated_risk_opportunity_neutral. If agreement is poor, keep exclusion.")
"""))

cells.append(md("""
### Why exclusion is the methodologically correct default
1. `distilroberta-base-climate-sentiment` was fine-tuned on **corporate disclosure paragraphs**, labels = risk / neutral / opportunity (financial/transition semantics).
2. The model card states it **may not perform well on sentences**.
3. Our units are ~10-token **news headlines**, a different genre and construct.
4. A climate-pretrained LM is not automatically valid for this task.

Existing exploratory title predictions under `OLD/analysis/titles_only/` are retained for reference but are not promoted into RQ1 claims here.
"""))

cells.append(md("""
## 8. Public opinion analysis (RQ2)

### CCAM constructs used
Exact items are documented in `analysis/ccam.py` (`SURVEY_ITEMS`). Outcomes are **not** collapsed into arbitrary indices.

| Construct | Variable | Note |
|---|---|---|
| Concern | `worry` | Somewhat/Very worried |
| Perceived harm | `harm_US`, `harm_personally` | Separate personal vs collective |
| Immediacy (not efficacy) | `when_harm_US` | Urgency-related |
| Discussion engagement | `discuss_GW` | Weak agency-related proxy only |
| Policy support | `reg_CO2_pollutant`, `fund_research`, `priority`, `priority_cleanenergy` | Separate series |
| Transition support | `transition_economy` | High missingness — report n |

Weights: `weight_wave`.  
Media alignment: features aggregated over the **90 days before** each wave’s month-start date. Empty windows stay **NaN**, not zero.
"""))

cells.append(code("""
for key, meta in SURVEY_ITEMS.items():
    print(f"- {key}: {meta.get('construct')} | high = {meta.get('high_definition')}")
    if meta.get("caveat"):
        print(f"  caveat: {meta['caveat']}")

ccam_raw = load_ccam(CCAM_CSV)
ccam = apply_recodes(ccam_raw)
wave = wave_level_outcomes(ccam)
print(f"Respondents: {len(ccam):,} | waves: {len(wave)}")
wave[
    [
        "wave",
        "wave_date",
        "n",
        "worry_high",
        "harm_US_high",
        "discuss_often",
        "support_CO2_reg",
        "n_support_transition",
    ]
].head(10)
"""))

cells.append(code("""
media_features = frame_cols + ["n_frames", "multi_frame"]
emo_merge_cols = [c for c in PRIMARY_EMOTIONS if c in emos.columns]
media_full = frames.merge(emos[["id"] + emo_merge_cols], on="id", how="left")
media_panel = media_full[media_full["in_fixed_panel"]].copy()
feat_cols = media_features + emo_merge_cols

win_full = media_window_features(media_full, wave["wave_date"], feat_cols, days=90)
win_panel = media_window_features(media_panel, wave["wave_date"], feat_cols, days=90)
win_full = win_full.add_prefix("media_")
win_panel = win_panel.add_prefix("panel_")

panel = pd.concat(
    [wave.reset_index(drop=True), win_full.reset_index(drop=True), win_panel.reset_index(drop=True)],
    axis=1,
)
panel.to_csv(OUTPUTS_DIR / "wave_panel_media_ccam.csv", index=False)
print(
    panel[
        ["wave", "wave_date", "n", "media_n_media", "panel_n_media", "worry_high", "media_frame_politics"]
    ].head()
)
print(f"Waves with zero media in 90d window (full): {(panel['media_n_media'] == 0).sum()}")
"""))

cells.append(code("""
fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

ax = axes[0]
ax.plot(
    panel["wave_date"],
    100 * panel["worry_high"],
    marker="o",
    color="#4C72B0",
    label="CCAM worry high (%)",
)
ax.set_ylabel("CCAM prevalence (%)")
ax2 = ax.twinx()
ax2.plot(
    panel["wave_date"],
    panel["media_frame_politics"],
    marker="s",
    color="#DD8452",
    label="Media politics (% headlines)",
)
ax2.plot(
    panel["wave_date"],
    panel["media_frame_solutions"],
    marker="^",
    color="#55A868",
    label="Media solutions (% headlines)",
)
ax.set_title("Aligned series: public worry vs media politics/solutions (90d pre-wave)")
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines + lines2, labels + labels2, loc="upper left", fontsize=8)

ax = axes[1]
ax.plot(
    panel["wave_date"],
    100 * panel["support_CO2_reg"],
    marker="o",
    color="#4C72B0",
    label="Support CO2 regulation (%)",
)
ax2 = ax.twinx()
ax2.plot(
    panel["wave_date"],
    panel["media_frame_economics"],
    marker="s",
    color="#C44E52",
    label="Media economics (% headlines)",
)
ax.set_title("Aligned series: CO2 policy support vs media economics language")
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines + lines2, labels + labels2, loc="best", fontsize=8)
ax.set_xlabel("Survey wave date")
plt.tight_layout()
plt.show()
"""))

cells.append(code("""
opinion_cols = [
    "worry_high",
    "harm_US_high",
    "harm_personally_high",
    "when_harm_soon",
    "discuss_often",
    "support_CO2_reg",
    "support_fund_research",
    "priority_high",
]
media_cols = [
    "media_frame_science",
    "media_frame_scientific_uncertainty",
    "media_frame_skepticism_denial",
    "media_frame_politics",
    "media_frame_economics",
    "media_frame_solutions",
    "media_fear",
]

corr = spearman_pair_table(panel, media_cols, opinion_cols)
corr["abs_rho"] = corr["spearman_rho"].abs()
corr_sorted = corr.sort_values("abs_rho", ascending=False)
print("Top Spearman associations (levels; not causal):")
display(corr_sorted.head(20))
corr.to_csv(OUTPUTS_DIR / "rq2_spearman_levels.csv", index=False)

mat_cols = [
    "media_frame_politics",
    "media_frame_economics",
    "media_frame_solutions",
    "media_frame_scientific_uncertainty",
    "media_fear",
    "worry_high",
    "harm_US_high",
    "discuss_often",
    "support_CO2_reg",
]
sub = panel[mat_cols].dropna()
print(f"Complete-case waves for correlation matrix: {len(sub)}")
cmat = sub.corr(method="spearman")
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(cmat, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax, vmin=-1, vmax=1)
ax.set_title("Spearman correlation matrix (wave-level media + opinion indicators)")
plt.tight_layout()
plt.show()
"""))

cells.append(code("""
diff_cols = mat_cols
d = panel[["wave_date"] + diff_cols].dropna().sort_values("wave_date").copy()
for c in diff_cols:
    d[f"d_{c}"] = d[c].diff()
d_media = [f"d_{c}" for c in diff_cols if c.startswith("media_")]
d_opin = [f"d_{c}" for c in diff_cols if not c.startswith("media_")]
corr_d = spearman_pair_table(d, d_media, d_opin)
print("Top Spearman associations on first differences:")
display(corr_d.sort_values("spearman_rho", key=lambda s: s.abs(), ascending=False).head(15))
corr_d.to_csv(OUTPUTS_DIR / "rq2_spearman_firstdiff.csv", index=False)

print(
    "Lagged comparisons: with ~30 overlapping waves, a single 1-wave lag exploratory "
    "check is defensible; richer distributed-lag models are not."
)
lag = panel[["wave_date", "worry_high", "media_frame_politics"]].dropna().sort_values("wave_date").copy()
lag["media_frame_politics_lag1"] = lag["media_frame_politics"].shift(1)
sub = lag.dropna()
if len(sub) >= 5:
    rho, p = stats.spearmanr(sub["media_frame_politics_lag1"], sub["worry_high"])
    print(f"Lag-1 media politics vs worry: rho={rho:.3f}, p={p:.3f}, n={len(sub)}")
"""))

cells.append(md("""
## 9. Robustness and sensitivity analysis

Comparisons implemented:
1. Full corpus vs fixed outlet panel
2. Primary vs alternative lexicon definitions
3. Outlet composition / headline-length checks
4. Automated vs human consensus (**pending labels**)
5. Primary vs broader MediaCloud query — **not available** (single query in exports)

Topic modeling remains omitted. ClimateBERT remains non-primary.
"""))

cells.append(code("""
cmp = compare_full_vs_panel(frames, frames[frames["in_fixed_panel"]], time_col="year")
display(cmp.head())

fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
pairs = [
    ("science", axes[0, 0]),
    ("scientific_uncertainty", axes[0, 1]),
    ("politics", axes[1, 0]),
    ("solutions", axes[1, 1]),
]
for name, ax in pairs:
    ax.plot(cmp["year"], cmp[f"full_frame_{name}"], marker="o", label="full")
    ax.plot(cmp["year"], cmp[f"panel_frame_{name}"], marker="s", label="panel")
    ax.set_title(name)
    ax.set_ylabel("% headlines")
    ax.legend(fontsize=8)
plt.suptitle("Robustness: full corpus vs fixed outlet panel", y=1.02)
plt.tight_layout()
plt.show()
cmp.to_csv(OUTPUTS_DIR / "robustness_full_vs_panel.csv", index=False)
"""))

cells.append(code("""
alt_cols = [f"alt_frame_{n}" for n in FRAME_NAMES]
ann_prev = prevalence_by(frames, "year", frame_cols)
alt_prev = prevalence_by(frames, "year", alt_cols)
merged = ann_prev.merge(alt_prev, on="year")

fig, ax = plt.subplots(figsize=(11, 4.5))
for n in ["politics", "economics", "solutions", "scientific_uncertainty"]:
    ax.plot(merged["year"], merged[f"frame_{n}"], ls="-", marker="o", label=f"primary {n}")
    ax.plot(merged["year"], merged[f"alt_frame_{n}"], ls="--", marker="x", label=f"alt {n}")
ax.set_ylabel("% headlines")
ax.set_title("Sensitivity: primary vs alternative lexicon definitions")
ax.legend(ncols=2, fontsize=8)
plt.tight_layout()
plt.show()
"""))

cells.append(code("""
ann2 = pd.read_csv(ANNOTATION_SAMPLE_PATH)
has_adj = ann2["adjudicated_primary_frame"].astype(str).str.len().gt(0).sum()
print(f"Adjudicated primary frames available: {has_adj} / {len(ann2)}")
if has_adj:
    scored_sample = ann2.merge(frames[["id"] + frame_cols], on="id", how="left")

    def auto_primary(row):
        hits = {n: row[f"frame_{n}"] for n in FRAME_NAMES}
        if sum(hits.values()) == 0:
            return "none_unclear"
        if sum(hits.values()) > 1:
            return "multi_auto"
        return max(hits, key=hits.get)

    scored_sample["auto_primary_simple"] = scored_sample.apply(auto_primary, axis=1)
    mask = scored_sample["adjudicated_primary_frame"].astype(str).str.len() > 0
    cm = confusion_against_consensus(
        scored_sample.loc[mask, "adjudicated_primary_frame"],
        scored_sample.loc[mask, "auto_primary_simple"],
    )
    display(cm)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Human adjudicated primary frame vs simple automated mapping")
    plt.tight_layout()
    plt.show()
else:
    print("Skipping human<->auto confusion matrix until adjudication is complete.")

print(
    "Broader vs primary query comparison: UNAVAILABLE "
    "(single MediaCloud query in repository)."
)

# Sensitivity: titles that themselves contain climate keywords
kw_mask = frames["title_original"].str.contains(
    r"climate|global warming|greenhouse|climate crisis|climate disaster",
    case=False,
    na=False,
)
kw_prev = prevalence_by(frames[kw_mask], "year", frame_cols)
full_prev = prevalence_by(frames, "year", frame_cols)
fig, ax = plt.subplots(figsize=(11, 4.5))
for n in ["politics", "solutions", "scientific_uncertainty"]:
    ax.plot(full_prev["year"], full_prev[f"frame_{n}"], ls="-", marker="o", label=f"all titles {n}")
    ax.plot(kw_prev["year"], kw_prev[f"frame_{n}"], ls="--", marker="x", label=f"keyword titles {n}")
ax.set_ylabel("% headlines")
ax.set_title(
    f"Sensitivity: all titles vs climate-keyword titles (n={kw_mask.sum():,} / {len(frames):,})"
)
ax.legend(ncols=2, fontsize=8)
plt.tight_layout()
plt.show()
"""))

cells.append(md("""
## 10. Findings and conclusions

Update this section after (a) reading the computed figures above and (b) completing human annotation. The cell below writes a method-safe draft that does **not** invent empirical claims beyond pipeline setup.
"""))

cells.append(code("""
print("=== Descriptive anchors (computed) ===")
print(f"Headlines analyzed (2008-2025): {len(frames):,}")
print(
    f"Outlets: {frames['outlet'].nunique()} | "
    f"Fixed panel outlets: {frames.loc[frames['in_fixed_panel'], 'outlet'].nunique()}"
)
print("Mean annual framing prevalence (full corpus):")
print(annual_full[frame_cols].mean().round(2))
print("CCAM waves joined:", len(panel))
print("Top level associations:")
display(corr_sorted.head(5))

findings_path = OUTPUTS_DIR / "FINDINGS_DRAFT.md"
findings_path.write_text(
    "\\n".join(
        [
            "# Findings draft (update after annotation)",
            "",
            "## RQ1",
            "- Report whether politics/economics/solutions prevalence rises relative to scientific_uncertainty on BOTH full corpus and fixed panel.",
            "- If full-corpus trends disappear on the panel, attribute them to archive composition, not narrative change.",
            "- Do not treat NRC fear presence as audience emotion.",
            "- Denial lexicon prevalence should be interpreted cautiously (rare + high false-positive risk).",
            "",
            "## RQ2",
            "- Report Spearman level and first-difference associations with n_waves and p-values.",
            "- Use associational language only.",
            "- Do not claim efficacy findings; CCAM lacks clean efficacy items.",
            "- Flag transition_economy for missingness.",
            "",
            "## Validation still required",
            "- Dual annotation of 200 headlines",
            "- Kappa + confusion matrices vs dictionaries",
            "",
        ]
    )
)
print("Wrote", findings_path)
"""))

cells.append(md("""
### Conclusions checklist
1. Answer RQ1 with annual/quarterly framing prevalence on **full** and **panel** corpora.
2. Treat emotional-language results as exploratory pending human tone labels.
3. Answer RQ2 with wave-aligned associations; emphasize uncertainty and small wave-N.
4. State headline-only and MediaCloud composition limits plainly.
5. List remaining manual tasks (`docs/REMAINING_TASKS.md`).

### Topic modeling decision (explicit)
BERTopic / embedding cluster methods are **omitted from primary analysis** because titles average ~10 tokens and do not reliably support stable, interpretable topic models for RQ claims.

### ClimateBERT decision (explicit)
Excluded as primary risk/opportunity measure for headlines; see Section 7.
"""))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    },
    "cells": cells,
}

OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {OUT} with {len(cells)} cells")
