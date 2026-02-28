"""
drift.py — Tracks quality and consistency shifts between baseline and current runs.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class DriftMetrics:
    consistency_drift: float   # Difference in consistency %
    risk_drift: float          # Difference in risk index
    confidence_drift: float    # Difference in confidence score
    is_regression: bool        # True if quality dropped significantly

class PromptDriftTracker:
    """
    Compares two ValidationReports or batch runs to detect drift.
    """

    def calculate_drift(self, current: dict, baseline: dict) -> DriftMetrics:
        # Consistency Drift
        curr_sim = current.get("consistency", {}).get("consistency_score", 0)
        base_sim = baseline.get("consistency", {}).get("consistency_score", 0)
        c_drift = curr_sim - base_sim

        # Hallucination Risk Drift (Lower risk is better, so base - curr)
        curr_risk = current.get("hallucination", [{}])[0].get("risk_index", 0)
        base_risk = baseline.get("hallucination", [{}])[0].get("risk_index", 0)
        r_drift = base_risk - curr_risk  # Positive = Improvement, Negative = Regression

        # Confidence Drift
        curr_conf = current.get("confidence", [{}])[0].get("score", 0)
        base_conf = baseline.get("confidence", [{}])[0].get("score", 0)
        conf_drift = curr_conf - base_conf

        # Logic for regression (e.g. >10% drop in consistency or >10 points drop in confidence)
        is_regression = (c_drift < -10) or (conf_drift < -10) or (r_drift < -10)

        return DriftMetrics(
            consistency_drift=round(c_drift, 2),
            risk_drift=round(r_drift, 2),
            confidence_drift=round(conf_drift, 2),
            is_regression=is_regression
        )

    def track_batch_drift(self, current_reports: List[dict], baseline_reports: List[dict]) -> Dict:
        """
        Compare two batches of reports.
        """
        if not baseline_reports:
            return {"status": "NO_BASELINE", "avg_drift": 0}

        total_drift = 0
        regressions = 0

        # Match by prompt if possible
        baseline_map = {r.get("prompt"): r for r in baseline_reports}
        
        for curr in current_reports:
            prompt = curr.get("prompt")
            base = baseline_map.get(prompt)
            if base:
                metrics = self.calculate_drift(curr, base)
                total_drift += metrics.confidence_drift
                if metrics.is_regression:
                    regressions += 1

        avg_drift = total_drift / len(current_reports) if current_reports else 0
        
        return {
            "status": "TRACKED",
            "avg_drift": round(avg_drift, 2),
            "regressions_found": regressions,
            "comparison_count": len(baseline_reports)
        }
