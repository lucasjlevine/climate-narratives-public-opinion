# How Climate Change Narratives Shape Public Opinion

Revised academic NLP notebook linking **MediaCloud US climate news headlines (2008–2025)** to **Yale/George Mason CCAM public opinion**.

## Start here

1. Open [`climate_narratives_public_opinion.ipynb`](climate_narratives_public_opinion.ipynb) (project root; use the project `.venv`).
2. Read [`docs/annotation_codebook.md`](docs/annotation_codebook.md) before labeling.
3. Label [`outputs/annotation_sample_200.csv`](outputs/annotation_sample_200.csv) with two independent annotators.
4. See [`docs/METHODOLOGY_CHANGES.md`](docs/METHODOLOGY_CHANGES.md) and [`docs/REMAINING_TASKS.md`](docs/REMAINING_TASKS.md).

Raw data inputs live under [`inputs/`](inputs/) (MediaCloud exports + CCAM). Prior exploratory code/artifacts remain under [`OLD/`](OLD/) and are not the primary path.

## Rebuild cached outputs

```bash
source .venv/bin/activate
python -m analysis.build_and_score
```

This rebuilds:
- `outputs/titles_corpus.parquet`
- `outputs/headline_frame_scores.parquet`
- `outputs/headline_emotion_scores.parquet`
- `outputs/annotation_sample_200.csv` (empty annotation fields; seed=42)

## Key methodological choices

| Choice | Reason |
|---|---|
| Titles only | Project scope; full archive coverage without scrape bias |
| Binary frame presence → % headlines | Headlines ≈ 10 tokens; hits/1k tokens unreliable |
| Split science / uncertainty / denial | Distinct constructs for RQ1 |
| Fixed 78-outlet panel | MediaCloud composition changes over time |
| No VADER primary | Disaster lexicon ≠ anti-climate framing |
| No BERTopic primary | Titles too short for defensible topics |
| No ClimateBERT primary | Trained on corporate paragraphs; model card warns on sentences |
| Dual human annotation required | Construct validation for dictionaries |

## MediaCloud query (all yearly exports)

`"climate change" OR "global warming" OR "climate disaster" OR "climate crisis"`  
Collection `34412234`. Matching can hit article bodies in the MediaCloud index, so **some titles lack climate keywords** even though the story entered the climate query result set. That is a headline-only analysis limitation, not a preprocessing bug.

## Layout

```
project/
├── climate_narratives_public_opinion.ipynb
├── analysis/                  # reusable Python modules
├── docs/                      # codebook, methodology notes, remaining tasks
├── inputs/                    # raw MediaCloud + CCAM data
│   ├── mediacloud/
│   └── ccam/
├── outputs/                   # corpus, scores, annotation sample
├── OLD/                       # prior notebooks and derived artifacts
└── .venv/
```
