# Human Annotation Codebook
## Project: How Climate Change Narratives Shape Public Opinion

**Status:** Codebook frozen *before* inspection of automated model outputs for the annotation sample.  
**Unit of analysis:** MediaCloud **article title / headline only** (not full text).  
**Sample:** 200 headlines, year-stratified with within-year outlet-diversity weights (`outputs/annotation_sample_200.csv`).  
**Seed:** 42  

---

## 1. Purpose

Human labels provide the construct-validity benchmark for dictionary framing scores and (optionally) for exploratory emotion / risk–opportunity classifications. Automated scores must **never** pre-fill annotation cells.

Annotators work **independently**. Adjudication happens only after both annotators finish.

---

## 2. General coding rules

1. Code what the **headline is mainly about**, not what a reader might feel.
2. A keyword appearing in a title is **not** sufficient by itself if the headline’s topic is clearly something else (irony, quotation of an opponent, sports metaphor, etc.).
3. **Impacts ≠ negative tone.** A headline about wildfires, floods, or extinctions can be an *impacts* frame and still be journalistically neutral.
4. **Debate / skeptical ≠ denial.** Do not code denial unless the headline asserts falsehood, hoax, scam, or that warming is not happening.
5. If two frames are present, choose the **primary** (dominant topic) and optionally a **secondary**.
6. If unclear, use `none_unclear` and explain in notes.
7. Do not use knowledge of later full-text articles; titles only.

---

## 3. Primary / secondary framing

Allowed labels:

| Label | Definition | Examples (illustrative) | Non-examples |
|---|---|---|---|
| `science` | Headline centers climate research, findings, scientific institutions, physical climate processes as knowledge claims | “NOAA: 2023 hottest year on record”; “IPCC warns…” | Political fight *about* a science report (often `politics`) |
| `scientific_uncertainty` | Incomplete knowledge, open questions, contested evidence **without** claiming climate change is fake | “Scientists still uncertain about cloud feedbacks”; “Skeptical of new study’s methods” | “Climate change is a hoax” → `skepticism_denial` |
| `skepticism_denial` | Asserts climate change / warming is fake, a hoax/scam, or that scientists are lying | “Global warming hoax exposed” | “Debate heats up in Senate” → usually `politics` |
| `politics` | Elections, parties, presidents, Congress, legislation fights, partisan conflict | “Biden climate bill faces GOP opposition” | Clean-energy *policy solution* without partisan conflict may be `solutions` |
| `economics` | Costs, jobs, markets, prices, industry profits, fossil-fuel business framing | “Climate rules could cost manufacturing jobs” | Carbon tax as a *policy instrument* may be `solutions` or `politics`—use judgment |
| `solutions` | Mitigation, adaptation, renewables, net-zero, named climate policies as actions | “Solar boom cuts emissions”; “Cities adapt to rising seas” | Pure disaster story with no response → `impacts` |
| `impacts` | Physical/social consequences (extremes, health, migration, ecosystems) without being primarily science-methods or politics | “Floods displace thousands amid warming” | “Fear grips coastal towns” alone is tone, not automatically impacts |
| `other` | Clearly on-topic climate headline that fits none of the above | Lifestyle, arts, sports metaphors tied to climate | — |
| `none_unclear` | Cannot tell from the title | Extremely vague or truncated titles | — |

**Secondary frame:** optional; use the same label set or leave blank.

---

## 4. Emotional language

`annotatorK_emotional_language` — free text using this scheme:

- `none` — no clear emotion lexicon beyond ordinary news wording
- one or more of: `fear`, `anger`, `hope`, `sadness`, `urgency`, `other:<word>`

Mark emotional **language in the headline**, not inferred audience feelings.

---

## 5. Tone (about climate issue presentation)

`negative` / `neutral` / `positive`

- Tone is about how climate change / response is **evaluated** in the headline wording.
- Describing a disaster impact can still be `neutral` journalism.
- Mockery of climate science → often `negative` *and* possibly `skepticism_denial`.
- Celebrating a climate solution → often `positive` *and* `solutions`.

---

## 6. Risk / opportunity / neutral

This is a **separate** judgment from emotional tone:

| Label | Meaning in this project |
|---|---|
| `risk` | Headline emphasizes climate hazards, damages, threats, dangers |
| `opportunity` | Headline emphasizes benefits, gains, upside of climate action or transition |
| `neutral` | Neither; descriptive, procedural, or mixed without clear valence of risk/opportunity |

Corporate ClimateBERT “risk/opportunity” labels are **not** the coding target here; use the definitions above for news headlines.

---

## 7. Workflow

1. Annotator 1 completes all rows for fields `annotator1_*`.
2. Annotator 2 completes `annotator2_*` without seeing Annotator 1.
3. Compute Cohen’s κ on primary frame (and separately on tone / risk-opportunity).
4. Adjudicate disagreements into `adjudicated_*`.
5. Only then compare consensus labels to automated dictionaries / optional models.

---

## 8. Agreement statistics

- **Two annotators:** Cohen’s kappa on primary frame; report percent agreement as well.
- **Three+ annotators:** Fleiss’ kappa.
- Interpret κ cautiously on imbalanced label distributions; show confusion matrices.

---

## 9. What not to do

- Do not paste model predictions into annotation columns.
- Do not treat MediaCloud missing years/outlets as “no coverage.”
- Do not code `skepticism_denial` from the words “debate” or “skeptical” alone.
- Do not equate impact coverage with anti-climate sentiment.
