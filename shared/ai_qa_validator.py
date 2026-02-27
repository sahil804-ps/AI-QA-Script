"""
ai_qa_validator.py — Core engine for AI Response Quality Validation

Three components:
  1. ConsistencyChecker  — Compares multiple AI responses for semantic similarity
  2. HallucinationDetector — Flags suspicious patterns in a response
  3. ConfidenceScorer — Scores response quality 0-100
"""

import re
from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from shared.config import (
    CONSISTENCY_THRESHOLD,
    OVERCONFIDENT_PHRASES,
    SUSPICIOUS_NUMERIC_PATTERNS,
    HEDGED_HALLUCINATION_PHRASES,
    HALLUCINATION_RISK_WEIGHTS,
    HALLUCINATION_RISK_LEVELS,
    CONFIDENCE_WEIGHTS,
    IDEAL_WORD_COUNT_RANGE,
    HEALTHY_HEDGE_PHRASES,
)


# ═══════════════════════════════════════════════════════════════
# Data Classes for Results
# ═══════════════════════════════════════════════════════════════

@dataclass
class ConsistencyResult:
    verdict: str                        # CONSISTENT | INCONSISTENT | NEEDS_MORE_SAMPLES
    average_similarity: float           # 0.0 – 1.0
    pairwise_scores: List[dict]         # [{pair: (i, j), score: float}]
    num_responses: int
    threshold_used: float

    def to_dict(self):
        return asdict(self)


@dataclass
class HallucinationMatch:
    category: str                       # overconfident | suspicious_numeric | hedged
    pattern: str
    matched_text: str
    sentence: str


@dataclass
class HallucinationResult:
    risk_level: str                     # LOW | MEDIUM | HIGH
    risk_score: int
    matches: List[HallucinationMatch]
    total_sentences: int
    flagged_sentences: int

    def to_dict(self):
        return asdict(self)


@dataclass
class ConfidenceResult:
    score: float                        # 0 – 100
    grade: str                          # A / B / C / D / F
    breakdown: dict                     # {component: score}
    word_count: int
    verdict: str                        # RELIABLE | ACCEPTABLE | UNRELIABLE

    def to_dict(self):
        return asdict(self)


# ═══════════════════════════════════════════════════════════════
# 1. Consistency Checker
# ═══════════════════════════════════════════════════════════════

class ConsistencyChecker:
    """
    Compares 2+ AI responses for the same prompt using TF-IDF cosine similarity.
    """

    def __init__(self, threshold: float = CONSISTENCY_THRESHOLD):
        self.threshold = threshold

    def check(self, responses: List[str]) -> ConsistencyResult:
        n = len(responses)

        if n < 2:
            return ConsistencyResult(
                verdict="NEEDS_MORE_SAMPLES",
                average_similarity=0.0,
                pairwise_scores=[],
                num_responses=n,
                threshold_used=self.threshold,
            )

        vectorizer = TfidfVectorizer(stop_words="english")
        try:
            tfidf_matrix = vectorizer.fit_transform(responses)
        except ValueError:
            return ConsistencyResult(
                verdict="NEEDS_MORE_SAMPLES",
                average_similarity=0.0,
                pairwise_scores=[],
                num_responses=n,
                threshold_used=self.threshold,
            )

        pairwise_scores = []
        scores = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(
                    cosine_similarity(tfidf_matrix[i], tfidf_matrix[j])[0][0]
                )
                pairwise_scores.append({"pair": (i + 1, j + 1), "score": round(sim, 4)})
                scores.append(sim)

        avg_sim = float(np.mean(scores)) if scores else 0.0
        verdict = "CONSISTENT" if avg_sim >= self.threshold else "INCONSISTENT"

        return ConsistencyResult(
            verdict=verdict,
            average_similarity=round(avg_sim, 4),
            pairwise_scores=pairwise_scores,
            num_responses=n,
            threshold_used=self.threshold,
        )


# ═══════════════════════════════════════════════════════════════
# 2. Hallucination Detector
# ═══════════════════════════════════════════════════════════════

class HallucinationDetector:
    """
    Pattern-based detection of hallucination signals in AI responses.
    """

    PATTERN_CATEGORIES = [
        ("overconfident",       OVERCONFIDENT_PHRASES),
        ("suspicious_numeric",  SUSPICIOUS_NUMERIC_PATTERNS),
        ("hedged",              HEDGED_HALLUCINATION_PHRASES),
    ]

    def detect(self, response: str) -> HallucinationResult:
        sentences = self._split_sentences(response)
        matches: List[HallucinationMatch] = []
        flagged_sentence_indices = set()

        for category, patterns in self.PATTERN_CATEGORIES:
            for pattern in patterns:
                for idx, sentence in enumerate(sentences):
                    found = re.findall(pattern, sentence, flags=re.IGNORECASE)
                    if found:
                        for match_text in found:
                            if isinstance(match_text, tuple):
                                match_text = " ".join(match_text)
                            matches.append(HallucinationMatch(
                                category=category,
                                pattern=pattern,
                                matched_text=match_text,
                                sentence=sentence.strip(),
                            ))
                        flagged_sentence_indices.add(idx)

        # Calculate risk score
        risk_score = 0
        for m in matches:
            risk_score += HALLUCINATION_RISK_WEIGHTS.get(m.category, 1)

        risk_level = "LOW"
        for level, (lo, hi) in HALLUCINATION_RISK_LEVELS.items():
            if lo <= risk_score <= hi:
                risk_level = level
                break

        return HallucinationResult(
            risk_level=risk_level,
            risk_score=risk_score,
            matches=matches,
            total_sentences=len(sentences),
            flagged_sentences=len(flagged_sentence_indices),
        )

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        # Simple sentence splitter on . ! ?
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in sentences if s]


# ═══════════════════════════════════════════════════════════════
# 3. Confidence Scorer
# ═══════════════════════════════════════════════════════════════

class ConfidenceScorer:
    """
    Assigns a quality confidence score (0–100) to an AI response.
    """

    def score(self, response: str) -> ConfidenceResult:
        words = response.split()
        word_count = len(words)
        breakdown = {}

        # 1. Length Score (0-20)
        low, high = IDEAL_WORD_COUNT_RANGE
        if word_count < low:
            # Too short — penalise
            ratio = word_count / low
            breakdown["length_score"] = round(ratio * CONFIDENCE_WEIGHTS["length_score"], 2)
        elif word_count > high:
            # Too long — slight penalty
            over = (word_count - high) / high
            penalty = min(over * 0.5, 1.0)
            breakdown["length_score"] = round((1 - penalty) * CONFIDENCE_WEIGHTS["length_score"], 2)
        else:
            breakdown["length_score"] = float(CONFIDENCE_WEIGHTS["length_score"])

        # 2. Structure Score (0-20) — rewards paragraphs, lists, headings
        structure_signals = [
            bool(re.search(r'\n', response)),                     # has newlines
            bool(re.search(r'^\s*[-*•]\s', response, re.M)),      # bullet list
            bool(re.search(r'^\s*\d+\.\s', response, re.M)),      # numbered list
            bool(re.search(r'#{1,3}\s', response)),                # markdown heading
            bool(re.search(r'[A-Z][^.!?]{20,}[.!?]', response)), # proper sentences
        ]
        structure_ratio = sum(structure_signals) / len(structure_signals)
        breakdown["structure_score"] = round(
            max(structure_ratio, 0.3) * CONFIDENCE_WEIGHTS["structure_score"], 2
        )

        # 3. Hedging Score (0-20) — healthy hedging = good epistemics
        hedge_count = sum(
            1 for phrase in HEALTHY_HEDGE_PHRASES
            if phrase.lower() in response.lower()
        )
        # 1-4 healthy hedges is ideal
        if hedge_count == 0:
            hedge_ratio = 0.4        # No hedging = potentially overconfident
        elif hedge_count <= 4:
            hedge_ratio = 1.0
        elif hedge_count <= 8:
            hedge_ratio = 0.7        # Some over-hedging
        else:
            hedge_ratio = 0.4        # Excessive uncertainty
        breakdown["hedging_score"] = round(hedge_ratio * CONFIDENCE_WEIGHTS["hedging_score"], 2)

        # 4. Clarity Score (0-20) — penalise ALL CAPS, excessive punctuation
        all_caps_words = len(re.findall(r'\b[A-Z]{4,}\b', response))
        punct_abuse = len(re.findall(r'[!?]{2,}', response))
        clarity_penalty = min((all_caps_words + punct_abuse) * 0.1, 1.0)
        breakdown["clarity_score"] = round(
            (1 - clarity_penalty) * CONFIDENCE_WEIGHTS["clarity_score"], 2
        )

        # 5. Completeness (0-20) — does response end properly?
        stripped = response.strip()
        ends_properly = bool(re.search(r'[.!?)\]"]$', stripped))
        not_truncated = word_count > 10
        breakdown["completeness"] = round(
            (0.6 if ends_properly else 0.2) * CONFIDENCE_WEIGHTS["completeness"] +
            (0.4 if not_truncated else 0.1) * CONFIDENCE_WEIGHTS["completeness"],
            2,
        )

        total_score = round(sum(breakdown.values()), 2)
        total_score = max(0.0, min(100.0, total_score))

        # Grade
        if total_score >= 85:
            grade, verdict = "A", "RELIABLE"
        elif total_score >= 70:
            grade, verdict = "B", "RELIABLE"
        elif total_score >= 55:
            grade, verdict = "C", "ACCEPTABLE"
        elif total_score >= 40:
            grade, verdict = "D", "UNRELIABLE"
        else:
            grade, verdict = "F", "UNRELIABLE"

        return ConfidenceResult(
            score=total_score,
            grade=grade,
            breakdown=breakdown,
            word_count=word_count,
            verdict=verdict,
        )


# ═══════════════════════════════════════════════════════════════
# Unified Validator
# ═══════════════════════════════════════════════════════════════

@dataclass
class ValidationReport:
    prompt: Optional[str]
    responses: List[str]
    consistency: Optional[ConsistencyResult]
    hallucination: List[HallucinationResult]
    confidence: List[ConfidenceResult]
    overall_verdict: str          # PASS | WARN | FAIL

    def to_dict(self):
        return {
            "prompt": self.prompt,
            "overall_verdict": self.overall_verdict,
            "num_responses": len(self.responses),
            "consistency": self.consistency.to_dict() if self.consistency else None,
            "hallucination": [h.to_dict() for h in self.hallucination],
            "confidence": [c.to_dict() for c in self.confidence],
        }


class AIQAValidator:
    """
    Orchestrates consistency, hallucination, and confidence checks.
    """

    def __init__(self):
        self.consistency_checker = ConsistencyChecker()
        self.hallucination_detector = HallucinationDetector()
        self.confidence_scorer = ConfidenceScorer()

    def validate(
        self,
        responses: List[str],
        prompt: Optional[str] = None,
        skip_consistency: bool = False,
    ) -> ValidationReport:

        # Consistency
        if skip_consistency or len(responses) < 2:
            consistency_result = None
        else:
            consistency_result = self.consistency_checker.check(responses)

        # Per-response checks
        hallucination_results = [
            self.hallucination_detector.detect(r) for r in responses
        ]
        confidence_results = [
            self.confidence_scorer.score(r) for r in responses
        ]

        # Overall verdict
        overall_verdict = self._compute_verdict(
            consistency_result, hallucination_results, confidence_results
        )

        return ValidationReport(
            prompt=prompt,
            responses=responses,
            consistency=consistency_result,
            hallucination=hallucination_results,
            confidence=confidence_results,
            overall_verdict=overall_verdict,
        )

    @staticmethod
    def _compute_verdict(
        consistency: Optional[ConsistencyResult],
        hallucinations: List[HallucinationResult],
        confidences: List[ConfidenceResult],
    ) -> str:
        flags = []

        if consistency and consistency.verdict == "INCONSISTENT":
            flags.append("inconsistent")

        if any(h.risk_level == "HIGH" for h in hallucinations):
            flags.append("high_hallucination")

        if any(c.verdict == "UNRELIABLE" for c in confidences):
            flags.append("unreliable_confidence")

        warnings = []
        if consistency and consistency.verdict == "INCONSISTENT":
            warnings.append("inconsistent")
        if any(h.risk_level == "MEDIUM" for h in hallucinations):
            warnings.append("medium_hallucination")
        if any(c.grade == "C" for c in confidences):
            warnings.append("low_confidence")

        if flags:
            return "FAIL"
        elif warnings:
            return "WARN"
        return "PASS"
