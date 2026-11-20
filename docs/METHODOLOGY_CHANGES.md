# Methodological changes and justifications

Revised notebook: `climate_narratives_public_opinion.ipynb`  
Prior materials preserved under `OLD/`.

## Summary of changes

| Prior approach (`OLD/`) | Revised approach | Justification |
|---|---|---|
| Full-text scrape + titles as sensitivity | **Titles only** as primary corpus | Project decision; scrape unevenness; titles cover full archive |
| Combined `science_uncertainty` lexicon | Split into **science**, **scientific_uncertainty**, **skepticism_denial** | RQ1 asks about shift from scientific uncertainty; denial is a different construct |
| Hits per 1,000 tokens | **Binary presence → % of headlines** | Mean headline length ≈ 10 tokens; rate-per-1k is unstable |
| VADER as primary polarity | **Not used as primary** | Disaster vocabulary conflated with negative framing of climate |
| NRC token-share via NRCLex frequencies | NRC **binary emotion presence** (% headlines) | Comparable to framing metric; still exploratory |
| BERTopic on full text / embeddings | **Omitted from primary analysis** | ~10.5-token headlines lack sufficient content for stable topics |
| ClimateBERT on title sample as Method C | **Excluded as primary**; suitability documented | Trained on corporate disclosure paragraphs; model card warns against sentences |
| Dominant-frame from max hit rate | Multi-label presence + human **primary frame** | Headlines often mix cues; human primary frame is the construct of interest |
| Weak archive composition checks | **Fixed outlet panel** (78 outlets present every year 2008–2025) + full corpus | MediaCloud source inventory changes; composition can fake trends |
| Annotation described as stretch / single reader | **Required**: codebook first, n=200 stratified sample, dual annotation fields, κ | Construct validation is essential for dictionary methods |
| Agency treated via discuss + immediacy proxies without enough caution | Same items retained but **explicitly not called efficacy**; no composite agency index | CCAM lacks clean efficacy items |

## Preserved useful work

- MediaCloud yearly CSV schema and query documentation (`*-content.txt`).
- CCAM wave-date parsing and weighted prevalence logic (rewritten in `analysis/ccam.py`).
- Recognition that news≠opinion and associations≠causation.
- ClimateBERT checkpoint inventory (reused for the exclusion decision).

## Study window

Primary analyses use **2008–2025**. A 2026 export exists (through 2026-11-04) but is incomplete and is not mixed into trend claims.

## Headline-only / query-match caveat

Empirically, only about **17%** of titles in the cleaned 2008–2025 corpus contain explicit climate/warming/greenhouse keywords. MediaCloud’s query can match story bodies in the index. The analysis therefore measures framing in titles of stories retrieved by a climate query, not exclusively in titles that mention climate. This weakens headline-only claims and should be stated in the paper.
