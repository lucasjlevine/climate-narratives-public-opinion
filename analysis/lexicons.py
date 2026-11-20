"""Custom climate framing lexicons for headline analysis.

Design principles
-----------------
1. Six conceptually distinct categories (RQ1), not a single science/uncertainty
   amalgam. Scientific uncertainty language is *not* the same as denial.
2. Headline scoring is **binary category presence** (1 if any term matches),
   then aggregated as the *percentage of headlines* containing each category.
   Hits per 1,000 tokens are inappropriate for ~10-token headlines.
3. Dictionaries are transparent and fixed before model-assisted labeling.
4. Ambiguous words are documented; presence is evidence of lexical cueing,
   not proof of a headline's primary frame.

Category definitions
--------------------
- science: reporting of climate science, research, institutions, findings.
- scientific_uncertainty: language about incomplete knowledge, open questions,
  contested evidence *without* asserting climate change is false.
- skepticism_denial: language asserting climate change is fake, a hoax, or
  that scientists are lying / there is no warming. Does NOT include bare
  "debate" or "skeptical" alone (those sit in scientific_uncertainty).
- politics: partisan actors, elections, legislation, administrations.
- economics: costs, jobs, markets, fossil-fuel industry economics.
- solutions: mitigation, adaptation, renewables, named climate policies.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

# ---------------------------------------------------------------------------
# Primary dictionaries (documented reasoning in docs/annotation_codebook.md)
# ---------------------------------------------------------------------------

FRAME_LEXICONS: dict[str, tuple[str, ...]] = {
    "science": (
        "ipcc",
        "nasa",
        "noaa",
        "peer reviewed",
        "peer-reviewed",
        "climate science",
        "climate scientist",
        "climate scientists",
        "study finds",
        "new study",
        "research shows",
        "research finds",
        "scientists say",
        "scientists find",
        "scientific study",
        "climate research",
        "atmospheric",
        "greenhouse gas",
        "greenhouse gases",
        "carbon dioxide",
        "co2 emissions",
        "temperature record",
        "global temperature",
        "climate model",
        "climate models",
        "paleoclimate",
        "ice core",
        "ice cores",
    ),
    "scientific_uncertainty": (
        # Incomplete knowledge / contested evidence — NOT denial
        "uncertainty",
        "uncertain",
        "unproven",
        "inconclusive",
        "contested",
        "disputed",
        "unresolved",
        "unknowns",
        "more research",
        "further research",
        "open question",
        "open questions",
        "scientific debate",
        "climate debate",  # debate about science; not automatically denial
        "debate among scientists",
        "lack of consensus",
        "unclear whether",
        # Bare "skeptical"/"skeptic" kept here: often means questioning
        # evidence strength, not asserting a hoax. Annotators must read context.
        "skeptic",
        "skeptics",
        "sceptical",
        "skeptical",
        "scepticism",
        "skepticism",
    ),
    "skepticism_denial": (
        # Explicit falsehood / conspiracy / rejection claims
        "hoax",
        "myth",
        "scam",
        "conspiracy",
        "fake science",
        "junk science",
        "climate alarmism",
        "climate alarmist",
        "climate alarmists",
        "climate hysteria",
        "denier",
        "deniers",
        "climate denial",
        "climate denier",
        "climate deniers",
        "global warming hoax",
        "climate change hoax",
        "not real",
        "isn't happening",
        "isnt happening",
        "climate lie",
        "climate lies",
        "settled science is wrong",
    ),
    "politics": (
        "congress",
        "senate",
        "house of representatives",
        "democrat",
        "democrats",
        "republican",
        "republicans",
        "partisan",
        "partisanship",
        "election",
        "elections",
        "campaign",
        "white house",
        "president",
        "presidential",
        "legislation",
        "legislators",
        "lawmakers",
        "bipartisan",
        "gop",
        "trump",
        "biden",
        "obama",
        "harris",
        "epa",
        "political",
        "politician",
        "politicians",
        "vote",
        "voters",
        "policy fight",
        "culture war",
        "executive order",
        "supreme court",
        "filibuster",
        "governor",
        "senators",
    ),
    "economics": (
        "cost",
        "costs",
        "expensive",
        "affordable",
        "price",
        "prices",
        "jobs",
        "job",
        "employment",
        "unemployment",
        "tax",
        "taxes",
        "carbon tax",
        "gdp",
        "economy",
        "economic",
        "industry",
        "industries",
        "market",
        "markets",
        "fossil fuel",
        "fossil fuels",
        "oil company",
        "oil companies",
        "coal industry",
        "subsidy",
        "subsidies",
        "investment",
        "investments",
        "consumer",
        "consumers",
        "business",
        "businesses",
        "profit",
        "profits",
        "recession",
        "inflation",
        "stock",
        "stocks",
        "wall street",
        "energy prices",
        "gas prices",  # retail fuel prices; ambiguous with atmospheric gas
    ),
    "solutions": (
        "solution",
        "solutions",
        "renewable",
        "renewables",
        "solar",
        "wind power",
        "wind energy",
        "clean energy",
        "clean power",
        "net zero",
        "net-zero",
        "carbon neutral",
        "carbon-neutral",
        "mitigation",
        "mitigate",
        "adaptation",
        "adapt",
        "energy transition",
        "decarboniz",
        "decarbonis",
        "electrification",
        "electric vehicle",
        "electric vehicles",
        "paris agreement",
        "climate action",
        "emissions reduction",
        "reduce emissions",
        "carbon capture",
        "green new deal",
        "inflation reduction act",
        "resilience",
        "resilient",
        "climate policy",
        "climate bill",
        "heat pump",
        "heat pumps",
        "battery storage",
    ),
}

FRAME_NAMES: tuple[str, ...] = tuple(FRAME_LEXICONS.keys())

# Alternative lexicon for sensitivity analysis: narrower science / broader politics
ALT_FRAME_LEXICONS: dict[str, tuple[str, ...]] = {
    "science": (
        "ipcc",
        "nasa",
        "noaa",
        "peer reviewed",
        "peer-reviewed",
        "climate science",
        "climate scientist",
        "climate scientists",
        "study finds",
        "new study",
        "research shows",
        "climate model",
        "climate models",
    ),
    "scientific_uncertainty": (
        "uncertainty",
        "uncertain",
        "unproven",
        "inconclusive",
        "scientific debate",
        "lack of consensus",
        "more research",
    ),
    "skepticism_denial": (
        "hoax",
        "myth",
        "scam",
        "conspiracy",
        "junk science",
        "climate alarmism",
        "climate denial",
        "climate denier",
        "climate deniers",
        "global warming hoax",
        "climate change hoax",
    ),
    "politics": FRAME_LEXICONS["politics"]
    + (
        "capitol hill",
        "state department",
        "interior department",
        "cabinet",
        "nominee",
        "confirmation hearing",
    ),
    "economics": (
        "cost",
        "costs",
        "jobs",
        "tax",
        "taxes",
        "carbon tax",
        "economy",
        "economic",
        "fossil fuel",
        "fossil fuels",
        "subsidy",
        "subsidies",
        "investment",
        "inflation",
        "energy prices",
        # intentionally drop bare "gas", "oil", "market", "price"
    ),
    "solutions": FRAME_LEXICONS["solutions"],
}


def _compile_patterns(lexicons: Mapping[str, tuple[str, ...]]) -> dict[str, re.Pattern[str]]:
    patterns: dict[str, re.Pattern[str]] = {}
    for name, terms in lexicons.items():
        ordered = sorted(set(terms), key=len, reverse=True)
        parts = [re.escape(t) for t in ordered]
        patterns[name] = re.compile(
            r"(?<![a-z0-9])(?:" + "|".join(parts) + r")(?![a-z0-9])",
            re.IGNORECASE,
        )
    return patterns


_PRIMARY_PATTERNS = _compile_patterns(FRAME_LEXICONS)
_ALT_PATTERNS = _compile_patterns(ALT_FRAME_LEXICONS)

_TOKEN_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def normalize_title(text: str) -> str:
    """Lowercase + collapse whitespace for matching; preserve original separately."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def binary_frame_presence(
    text: str,
    patterns: Mapping[str, re.Pattern[str]] | None = None,
) -> dict[str, int]:
    """Return 0/1 indicators for each frame category."""
    pats = patterns or _PRIMARY_PATTERNS
    if not (text or "").strip():
        return {name: 0 for name in pats}
    return {name: int(bool(pat.search(text))) for name, pat in pats.items()}


def matched_terms(text: str, frame: str, lexicons: Mapping[str, tuple[str, ...]] | None = None) -> list[str]:
    """Return which dictionary terms matched (for inspection / adjudication)."""
    lex = lexicons or FRAME_LEXICONS
    terms = lex[frame]
    ordered = sorted(set(terms), key=len, reverse=True)
    found = []
    for t in ordered:
        pat = re.compile(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", re.IGNORECASE)
        if pat.search(text or ""):
            found.append(t)
    return found


def score_titles_binary(
    titles: Iterable[str],
    *,
    use_alt: bool = False,
) -> list[dict[str, int]]:
    patterns = _ALT_PATTERNS if use_alt else _PRIMARY_PATTERNS
    return [binary_frame_presence(t, patterns) for t in titles]


# Documented ambiguous / high-false-positive terms
AMBIGUOUS_TERMS_NOTES: dict[str, str] = {
    "debate": (
        "May refer to policy debate, scientific debate, or televised debate; "
        "not coded as denial in the primary lexicon (only appears in scientific_uncertainty "
        "via multi-word phrases like 'scientific debate' / 'climate debate')."
    ),
    "skeptical / skeptic": (
        "Often means questioning evidence strength. Kept under scientific_uncertainty, "
        "not skepticism_denial. Human annotators must not equate skepticism with denial."
    ),
    "gas": (
        "Dropped from the primary economics list as a bare token (atmospheric gas vs. fuel). "
        "Retained only in 'gas prices' / fossil-fuel phrases."
    ),
    "oil": (
        "Bare 'oil' dropped from primary economics to reduce false hits "
        "(e.g., oil paintings, cooking oil). Prefer 'oil company/companies'."
    ),
    "ira": (
        "Dropped: often means Individual Retirement Account, not Inflation Reduction Act. "
        "Use the full act name instead."
    ),
    "transition": (
        "Bare 'transition' dropped (too broad). Prefer 'energy transition'."
    ),
    "evidence": (
        "Dropped from science list: too generic in news headlines."
    ),
    "adapt / adaptation": (
        "Can refer to evolutionary or business adaptation; retained because "
        "climate-adaptation is a core solutions frame, but presence ≠ primary frame."
    ),
}
