"""
config.py — Central configuration for AI QA Validator
"""

# ──────────────────────────────────────────────
# Consistency Checker Settings
# ──────────────────────────────────────────────
CONSISTENCY_THRESHOLD = 0.40  # Cosine similarity score below this = INCONSISTENT
# Note: TF-IDF cosine similarity for paraphrased text typically ranges 0.3–0.6.
# Use 0.40 as default (natural language). Raise to 0.70+ only for near-exact copies.

# ──────────────────────────────────────────────
# Hallucination Detector — Pattern Lists
# ──────────────────────────────────────────────

# Phrases that signal overconfidence / absolute claims
OVERCONFIDENT_PHRASES = [
    r"\balways\b",
    r"\bnever\b",
    r"\beveryone knows\b",
    r"\bproven fact\b",
    r"\bscientifically proven\b",
    r"\bstudies (show|prove|confirm)\b",
    r"\bwithout (a )?doubt\b",
    r"\b100\s*%\b",
    r"\babsolutely\b",
    r"\bcertainly\b",
    r"\bundeniably\b",
    r"\bguaranteed\b",
    r"\bimpossible\b",
    r"\bno one (can|could|will)\b",
    r"\bthe (only|best|worst)\b",
    r"\bfact is\b",
    r"\bits a fact\b",
    r"\bwithout question\b",
    r"\bcompletely certain\b",
    r"\bundisputed\b",
    r"\buniversally acknowledged\b",
    r"\bthere is no doubt that\b",
    r"\bcompletely (true|accurate|reliable)\b",
    r"\bfactually (correct|accurate)\b",
]

# Patterns that suggest fabricated numeric stats
SUSPICIOUS_NUMERIC_PATTERNS = [
    r"\b\d{1,3}(\.\d+)?\s*%\s*(of (people|users|experts|researchers|studies))\b",
    r"\b(according to|based on)\s+.*\bstudy\b",
    r"\b\d+\s*(million|billion|trillion)\s*(people|users|dollars)\b",
    r"\b(research|statistics|data)\s+show(s)?\b",
    r"\breach(es|ed)?\s+\d{1,3}(\.\d+)?\s*%\b",
    r"\bin the year \d{4}\b",
    r"\b(over|more than|approximately)\s+\d+\s+%\b",
    r"\bsurvey\s+of\s+\d+\s+participants\b",
]

# Hedged hallucination markers (AI guessing with false confidence)
HEDGED_HALLUCINATION_PHRASES = [
    r"\bI (believe|think|assume) (the|that) .{0,60}(is|are|was|were)\b",
    r"\bif I recall correctly\b",
    r"\bI('m| am) pretty sure\b",
    r"\bI might be wrong but\b",
    r"\bto the best of my knowledge\b",
    r"\bI cannot verify this but\b",
    r"\bI don't have real-time access but\b",
    r"\bI suspect that\b",
    r"\bit's likely that\s*.*\s*but I can't be certain\b",
]

# Risk scoring weights
HALLUCINATION_RISK_WEIGHTS = {
    "overconfident": 2,       # per match
    "suspicious_numeric": 3,  # per match
    "hedged": 1,              # per match
}

HALLUCINATION_RISK_LEVELS = {
    "LOW": (0, 2),
    "MEDIUM": (3, 6),
    "HIGH": (7, 9999),
}

# ──────────────────────────────────────────────
# Confidence Scorer Settings
# ──────────────────────────────────────────────

CONFIDENCE_WEIGHTS = {
    "length_score":     20,   # Penalise too-short or absurdly long responses
    "structure_score":  20,   # Rewards lists, paragraphs, headings
    "hedging_score":    20,   # Appropriate uncertainty language is healthy
    "clarity_score":    20,   # Punish excessive jargon / all-caps
    "completeness":     20,   # Response ends properly, not mid-sentence
}

IDEAL_WORD_COUNT_RANGE = (30, 600)

HEALTHY_HEDGE_PHRASES = [
    "I think", "I believe", "it seems", "it appears",
    "arguably", "generally", "typically", "often", "sometimes",
    "may", "might", "could", "possibly", "likely",
    "in most cases", "one perspective", "it depends",
]
