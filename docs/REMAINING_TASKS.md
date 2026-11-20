# Remaining data requirements and manual tasks

## Required manual work

1. **Dual human annotation** of `outputs/annotation_sample_200.csv` using `docs/annotation_codebook.md`.
   - Annotator 1 and Annotator 2 must work independently.
   - Fill only `annotator1_*` / `annotator2_*` first; adjudicate afterward.
2. **Compute inter-annotator agreement** (Cohen’s κ for two coders) after both finish; then fill `adjudicated_*`.
3. **Compare consensus labels** to automated `frame_*` indicators (and optional exploratory models) — notebook Section 4/9 cells are ready once labels exist.
4. **Close reading** of years/quarters with large framing shifts (especially where full corpus and fixed panel diverge).

## Data gaps / unavailable comparisons

| Desired item | Status |
|---|---|
| Broader vs primary MediaCloud query corpora | **Unavailable.** All yearly exports use the same query: `"climate change" OR "global warming" OR "climate disaster" OR "climate crisis"` (collection `34412234`). Robustness item deferred unless a second export is obtained. |
| Clean CCAM personal/collective efficacy items | **Unavailable** in `CCAM_SPSS_Data_2008-2025.csv`. Use documented proxies only (`discuss_GW`, `when_harm_US`) without calling them efficacy. |
| Article body text for primary analysis | Intentionally out of scope (titles only). Scraped text under `inputs/mediacloud/scraped/` remains available for optional sensitivity work. |
| Complete 2026 calendar year | Export ends 2026-11-04; exclude from primary 2008–2025 trends. |
| Second MediaCloud collection / outlet universe documentation beyond counts | Collection ID known; deeper inventory change logs not in-repo. |
| Title-only subset restricted to climate-keyword headlines | Optional sensitivity: repeat framing trends on the ~17% of titles with explicit climate keywords (or re-export MediaCloud with title-only matching if available). |

## Optional follow-ups (not required for minimum defensible paper)

- After adjudication, run ClimateBERT sentiment **only as exploratory** on the 200 labeled titles; report P/R/F1 + confusion matrix; do not promote to primary RQ1 evidence if agreement is weak.
- Obtain an alternate MediaCloud query (e.g., adding `greenhouse gas` / `carbon emissions`) for query-sensitivity analysis.
- Second-reader adjudication meeting notes for the paper appendix.
