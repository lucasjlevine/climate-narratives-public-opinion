"""Human annotation sample design and agreement helpers.

IMPORTANT: The codebook must be written *before* inspecting automated model
outputs. Annotation fields are exported empty — never pre-filled with model
predictions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analysis.paths import RANDOM_SEED

# Empty fields for human coders (two independent annotators + adjudication)
ANNOTATION_FIELDS = [
    "annotator1_primary_frame",
    "annotator1_secondary_frame",
    "annotator1_emotional_language",  # yes/no + which
    "annotator1_tone",  # negative / neutral / positive (about climate issue framing)
    "annotator1_risk_opportunity_neutral",
    "annotator1_notes",
    "annotator2_primary_frame",
    "annotator2_secondary_frame",
    "annotator2_emotional_language",
    "annotator2_tone",
    "annotator2_risk_opportunity_neutral",
    "annotator2_notes",
    "adjudicated_primary_frame",
    "adjudicated_secondary_frame",
    "adjudicated_emotional_language",
    "adjudicated_tone",
    "adjudicated_risk_opportunity_neutral",
    "adjudication_notes",
]

PRIMARY_FRAME_LABELS = [
    "science",
    "scientific_uncertainty",
    "skepticism_denial",
    "politics",
    "economics",
    "solutions",
    "impacts",  # climate impacts/disaster consequences without being "negative tone"
    "other",
    "none_unclear",
]


@dataclass
class SampleDesign:
    n_target: int
    seed: int
    n_years: int
    base_per_year: int
    remainder_years: list[int]
    n_drawn: int
    n_outlets_in_sample: int
    year_allocation: dict[int, int]


def stratified_headline_sample(
    corpus: pd.DataFrame,
    *,
    n: int = 200,
    seed: int = RANDOM_SEED,
    year_min: int | None = None,
    year_max: int | None = None,
) -> tuple[pd.DataFrame, SampleDesign, pd.DataFrame]:
    """Year-stratified random sample with outlet diversity constraints.

    Allocation
    ----------
    - Equal base allocation across years in the study window.
    - Remainder years receive +1 by deterministic round-robin from seed order.
    - Within each year, sample without replacement.
    - Soft outlet diversity: within-year sampling uses probability weights
      inversely related to outlet frequency that year (capped), so large wires
      do not dominate while small outlets remain includable.

    Inclusion probability for unit i in year y with within-year weight w_i:
        P(i selected | year y) ≈ (n_y * w_i) / sum_j w_j
    for sampling without replacement this is approximate; we record the
    design weights used for transparency.
    """
    df = corpus.copy()
    if year_min is not None:
        df = df[df["year"] >= year_min]
    if year_max is not None:
        df = df[df["year"] <= year_max]
    df = df.sort_values(["year", "id"]).reset_index(drop=True)

    years = sorted(df["year"].unique().tolist())
    n_years = len(years)
    if n_years == 0:
        raise ValueError("No rows available for sampling")
    if n < n_years:
        raise ValueError(f"n={n} smaller than number of years={n_years}")

    base = n // n_years
    rem = n % n_years
    rng = np.random.default_rng(seed)
    # Remainder distributed to a random permutation of years (reproducible)
    rem_years = rng.permutation(years)[:rem].tolist() if rem else []
    allocation = {y: base + (1 if y in rem_years else 0) for y in years}

    parts: list[pd.DataFrame] = []
    weight_rows: list[dict] = []
    for y in years:
        g = df[df["year"] == y].copy()
        n_y = allocation[y]
        n_y = min(n_y, len(g))
        # Inverse-frequency weights (sqrt dampening); floor so rare outlets aren't extreme
        outlet_counts = g["outlet"].value_counts()
        g["outlet_n_year"] = g["outlet"].map(outlet_counts)
        g["design_weight"] = 1.0 / np.sqrt(g["outlet_n_year"].clip(lower=1))
        # Convert to sampling probabilities
        p = g["design_weight"].to_numpy(dtype=float)
        p = p / p.sum()
        idx = rng.choice(g.index.to_numpy(), size=n_y, replace=False, p=p)
        take = g.loc[idx].copy()
        take["sample_year_allocation"] = allocation[y]
        take["inclusion_prob_approx"] = take["design_weight"] * n_y / g["design_weight"].sum()
        parts.append(take)
        for _, row in take.iterrows():
            weight_rows.append(
                {
                    "id": row["id"],
                    "year": y,
                    "outlet": row["outlet"],
                    "design_weight": row["design_weight"],
                    "inclusion_prob_approx": row["inclusion_prob_approx"],
                    "n_year_population": len(g),
                    "n_year_sample": n_y,
                }
            )

    sample = pd.concat(parts, ignore_index=True)
    sample = sample.sort_values(["year", "outlet", "id"]).reset_index(drop=True)
    sample.insert(0, "annotation_id", [f"ANN-{i:04d}" for i in range(1, len(sample) + 1)])

    design = SampleDesign(
        n_target=n,
        seed=seed,
        n_years=n_years,
        base_per_year=base,
        remainder_years=sorted(rem_years),
        n_drawn=len(sample),
        n_outlets_in_sample=int(sample["outlet"].nunique()),
        year_allocation=allocation,
    )
    weights = pd.DataFrame(weight_rows)
    return sample, design, weights


def export_labeling_spreadsheet(sample: pd.DataFrame, path) -> pd.DataFrame:
    """Export titles + empty annotation fields (no model predictions)."""
    cols = [
        "annotation_id",
        "id",
        "publish_date",
        "year",
        "outlet",
        "title_original",
        "url",
        "inclusion_prob_approx",
    ]
    out = sample[cols].copy()
    for field in ANNOTATION_FIELDS:
        out[field] = ""
    out.to_csv(path, index=False)
    return out


def cohens_kappa(y1: pd.Series, y2: pd.Series) -> float:
    """Cohen's kappa for two annotators (nominal labels)."""
    mask = y1.notna() & y2.notna() & (y1.astype(str).str.len() > 0) & (y2.astype(str).str.len() > 0)
    a = y1[mask].astype(str).to_numpy()
    b = y2[mask].astype(str).to_numpy()
    if len(a) == 0:
        return float("nan")
    labels = sorted(set(a) | set(b))
    idx = {lab: i for i, lab in enumerate(labels)}
    mat = np.zeros((len(labels), len(labels)), dtype=float)
    for x, y in zip(a, b):
        mat[idx[x], idx[y]] += 1
    n = mat.sum()
    po = np.trace(mat) / n
    pe = (mat.sum(axis=0) * mat.sum(axis=1)).sum() / (n**2)
    if pe == 1:
        return 1.0
    return float((po - pe) / (1 - pe))


def fleiss_kappa(ratings: pd.DataFrame) -> float:
    """Fleiss' kappa for ≥3 annotators; columns are annotators, rows are items."""
    # Convert to category counts per item
    cats = sorted({v for col in ratings.columns for v in ratings[col].dropna().astype(str).unique() if v})
    if not cats:
        return float("nan")
    n_items = len(ratings)
    n_raters = ratings.shape[1]
    counts = np.zeros((n_items, len(cats)), dtype=float)
    cat_idx = {c: i for i, c in enumerate(cats)}
    for i, (_, row) in enumerate(ratings.iterrows()):
        for v in row.astype(str):
            if v in cat_idx:
                counts[i, cat_idx[v]] += 1
    # Agreement
    P_i = ((counts**2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = P_i.mean()
    p_j = counts.sum(axis=0) / (n_items * n_raters)
    P_e = (p_j**2).sum()
    if P_e == 1:
        return 1.0
    return float((P_bar - P_e) / (1 - P_e))


def confusion_against_consensus(
    human: pd.Series,
    auto: pd.Series,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Confusion matrix: rows = human consensus, cols = automated label."""
    mask = human.notna() & auto.notna()
    h = human[mask].astype(str)
    a = auto[mask].astype(str)
    if labels is None:
        labels = sorted(set(h) | set(a))
    return pd.crosstab(h, a, rownames=["human"], colnames=["auto"]).reindex(
        index=labels, columns=labels, fill_value=0
    )
