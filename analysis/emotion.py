"""NRC Word-Emotion Association Lexicon helpers (exploratory).

Interpretation limits
---------------------
- Lexical association ≠ reader emotion.
- Disaster vocabulary (flood, wildfire, death) inflates fear/sadness without
  implying that the headline *frames climate change negatively as a concept*.
- Validate against human emotional-language annotations when available.
- VADER is intentionally not used as a primary tone method here.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

from analysis.lexicons import tokenize

NRC_EMOTIONS = (
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "trust",
)

# Primary exploratory set most relevant to climate news tone
PRIMARY_EMOTIONS = ("fear", "anger", "trust", "anticipation", "sadness")


@lru_cache(maxsize=1)
def load_nrc_lexicon() -> dict[str, set[str]]:
    """word -> set of emotion labels from the NRCLex English JSON."""
    try:
        import nrclex

        path = Path(nrclex.__file__).parent / "data" / "nrc_en.json"
    except Exception as exc:  # pragma: no cover
        raise FileNotFoundError("nrclex package with data/nrc_en.json is required") from exc

    raw = json.loads(path.read_text(encoding="utf-8"))
    return {w.lower(): set(labels) for w, labels in raw.items()}


def emotion_presence(text: str, lexicon: dict[str, set[str]] | None = None) -> dict[str, int]:
    """Binary: does the headline contain ≥1 NRC word linked to each emotion?"""
    lex = lexicon or load_nrc_lexicon()
    tokens = tokenize(text or "")
    present = {e: 0 for e in NRC_EMOTIONS}
    present["positive"] = 0
    present["negative"] = 0
    for tok in tokens:
        labels = lex.get(tok)
        if not labels:
            continue
        for lab in labels:
            if lab in present:
                present[lab] = 1
    return present


def score_titles_emotions(titles: Iterable[str]) -> list[dict[str, int]]:
    lex = load_nrc_lexicon()
    return [emotion_presence(t, lex) for t in titles]
