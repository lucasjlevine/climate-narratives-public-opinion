"""ClimateBERT suitability review for short news headlines.

Conclusion used by the notebook
-------------------------------
The publicly documented ClimateBERT sentiment head
(`climatebert/distilroberta-base-climate-sentiment`) is **not appropriate as a
primary measure of risk/opportunity framing in ~10-token news headlines**.

Evidence
--------
1. Checkpoint role: `distilroberta-base-climate-f` is a masked LM further
   pretrained on climate-related paragraphs (research abstracts, corporate /
   general news, company reports). It is not itself a classifier.
2. Sentiment head training data: expert-annotated *paragraphs from corporate
   annual and sustainability reports* (climatebert/climate_sentiment;
   train n≈1,000), labels = risk / neutral / opportunity.
3. Model card warning: "This model is trained on paragraphs. It may not
   perform well on sentences."
4. Construct mismatch: corporate disclosure "risk/opportunity" is financial /
   transition-risk language, not news-headline emotional tone or public-facing
   climate framing.
5. Domain mismatch: headlines average ~10.5 tokens; training examples are
   multi-sentence disclosure paragraphs.

Policy for this project
-----------------------
- Do **not** treat ClimateBERT labels as the RQ1 risk/opportunity series.
- Optional exploratory comparison against human risk/opportunity/neutral
  annotations is allowed *after* human coding is complete, with precision /
  recall / F1 and a confusion matrix reported honestly.
- If human labels disagree strongly, exclude the model from conclusions.
"""

from __future__ import annotations

from dataclasses import dataclass

CLIMATEBERT_SENTIMENT = "climatebert/distilroberta-base-climate-sentiment"
CLIMATEBERT_COMMITMENT = "climatebert/distilroberta-base-climate-commitment"
CLIMATEBERT_DETECTOR = "climatebert/distilroberta-base-climate-detector"
CLIMATEBERT_BASE_LM = "climatebert/distilroberta-base-climate-f"


@dataclass(frozen=True)
class SuitabilityDecision:
    use_as_primary: bool
    reason: str
    checkpoint: str
    training_domain: str
    label_set: tuple[str, ...]


DECISION = SuitabilityDecision(
    use_as_primary=False,
    reason=(
        "Fine-tuned on corporate disclosure paragraphs for financial "
        "risk/opportunity/neutral; model card warns against sentence-level use; "
        "headlines (~10 tokens) are out of domain for this construct."
    ),
    checkpoint=CLIMATEBERT_SENTIMENT,
    training_domain="corporate annual/sustainability report paragraphs",
    label_set=("risk", "neutral", "opportunity"),
)


def decision_summary() -> dict:
    return {
        "use_as_primary": DECISION.use_as_primary,
        "checkpoint": DECISION.checkpoint,
        "training_domain": DECISION.training_domain,
        "labels": list(DECISION.label_set),
        "reason": DECISION.reason,
        "base_lm": CLIMATEBERT_BASE_LM,
        "related_heads": {
            "commitment": CLIMATEBERT_COMMITMENT,
            "detector": CLIMATEBERT_DETECTOR,
        },
    }
