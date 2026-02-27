"""
report.py — Pretty console output + JSON report generator
"""

import json
from dataclasses import asdict
from typing import Optional

from colorama import init, Fore, Style

from ai_qa_validator import ValidationReport, ConsistencyResult, HallucinationResult, ConfidenceResult

init(autoreset=True)  # Colorama auto-reset


# ─── Colour helpers ───────────────────────────────────────────

def _col(text: str, color) -> str:
    return f"{color}{text}{Style.RESET_ALL}"


def _verdict_color(verdict: str) -> str:
    return {
        "PASS": Fore.GREEN,
        "WARN": Fore.YELLOW,
        "FAIL": Fore.RED,
        "CONSISTENT": Fore.GREEN,
        "INCONSISTENT": Fore.RED,
        "NEEDS_MORE_SAMPLES": Fore.YELLOW,
        "RELIABLE": Fore.GREEN,
        "ACCEPTABLE": Fore.YELLOW,
        "UNRELIABLE": Fore.RED,
        "LOW": Fore.GREEN,
        "MEDIUM": Fore.YELLOW,
        "HIGH": Fore.RED,
    }.get(verdict, Fore.WHITE)


def _badge(verdict: str) -> str:
    icons = {
        "PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌",
        "CONSISTENT": "✅", "INCONSISTENT": "❌", "NEEDS_MORE_SAMPLES": "⚠️ ",
        "RELIABLE": "✅", "ACCEPTABLE": "⚠️ ", "UNRELIABLE": "❌",
        "LOW": "✅", "MEDIUM": "⚠️ ", "HIGH": "🚨",
        "A": "🌟", "B": "👍", "C": "😐", "D": "👎", "F": "💀",
    }
    return icons.get(verdict, "•")


# ─── Section Printers ─────────────────────────────────────────

def _divider(char="═", width=60):
    print(_col(char * width, Fore.CYAN))


def _header(title: str):
    _divider()
    print(_col(f"  {title}", Fore.CYAN + Style.BRIGHT))
    _divider()


def print_consistency(result: Optional[ConsistencyResult]):
    if result is None:
        print(_col("  ⏭  Consistency check skipped (single response mode)", Fore.YELLOW))
        return

    _header("🔁  CONSISTENCY CHECK")
    vc = _verdict_color(result.verdict)
    print(f"  Verdict         : {_col(f'{_badge(result.verdict)} {result.verdict}', vc)}")
    print(f"  Avg Similarity  : {_col(f'{result.average_similarity:.4f}', Fore.WHITE)}  "
          f"(threshold: {result.threshold_used})")
    print(f"  Responses       : {result.num_responses}")

    if result.pairwise_scores:
        print()
        print(f"  {'Pair':<12} {'Score':<10} {'Status'}")
        print(f"  {'─'*12} {'─'*10} {'─'*12}")
        for pw in result.pairwise_scores:
            i, j = pw["pair"]
            sc = pw["score"]
            status = "✅ Similar" if sc >= result.threshold_used else "❌ Different"
            print(f"  R{i}  ↔  R{j}     {sc:<10.4f} {status}")
    print()


def print_hallucination(results: list, response_labels: list):
    _header("🚨  HALLUCINATION DETECTION")
    for label, result in zip(response_labels, results):
        vc = _verdict_color(result.risk_level)
        print(f"  {label}  Risk: {_col(f'{_badge(result.risk_level)} {result.risk_level}', vc)}"
              f"  |  Score: {result.risk_score}"
              f"  |  Flagged: {result.flagged_sentences}/{result.total_sentences} sentences")

        if result.matches:
            shown = {}
            for m in result.matches:
                key = m.sentence[:80]
                if key not in shown:
                    shown[key] = m.category
                    cat_color = {
                        "overconfident": Fore.RED,
                        "suspicious_numeric": Fore.MAGENTA,
                        "hedged": Fore.YELLOW,
                    }.get(m.category, Fore.WHITE)
                    print(f"    ↳ [{_col(m.category.upper(), cat_color)}] \"{m.sentence.strip()[:100]}\"")
        else:
            print(f"    ↳ No suspicious patterns detected.")
        print()


def print_confidence(results: list, response_labels: list):
    _header("📊  CONFIDENCE SCORES")
    for label, result in zip(response_labels, results):
        vc = _verdict_color(result.verdict)
        grade_col = _verdict_color(result.grade)
        print(f"  {label}  Score: {_col(f'{result.score}/100', Fore.WHITE)}"
              f"  Grade: {_col(f'{_badge(result.grade)} {result.grade}', grade_col)}"
              f"  Verdict: {_col(result.verdict, vc)}"
              f"  Words: {result.word_count}")
        print(f"    Breakdown:")
        for component, val in result.breakdown.items():
            bar_len = int(val / 20 * 15)
            bar = "█" * bar_len + "░" * (15 - bar_len)
            print(f"      {component:<20}: {_col(bar, Fore.CYAN)} {val:.1f}/20")
        print()


def print_overall(verdict: str, prompt: Optional[str]):
    _divider("═")
    vc = _verdict_color(verdict)
    print()
    if prompt:
        print(f"  Prompt: \"{prompt[:80]}{'...' if len(prompt) > 80 else ''}\"")
    print(f"  Overall Verdict : {_col(f'{_badge(verdict)}  {verdict}', vc + Style.BRIGHT)}")
    print()
    _divider("═")


def print_report(report: ValidationReport):
    print()
    print(_col("╔══════════════════════════════════════════════════════════╗", Fore.CYAN + Style.BRIGHT))
    print(_col("║          🤖  AI RESPONSE QUALITY VALIDATOR               ║", Fore.CYAN + Style.BRIGHT))
    print(_col("╚══════════════════════════════════════════════════════════╝", Fore.CYAN + Style.BRIGHT))
    print()

    labels = [f"[Response {i+1}]" for i in range(len(report.responses))]

    print_consistency(report.consistency)
    print_hallucination(report.hallucination, labels)
    print_confidence(report.confidence, labels)
    print_overall(report.overall_verdict, report.prompt)


# ─── JSON Export ──────────────────────────────────────────────

def _result_to_dict(report: ValidationReport) -> dict:
    def _serialize(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return {k: _serialize(v) for k, v in asdict(obj).items()}
        if isinstance(obj, list):
            return [_serialize(i) for i in obj]
        if isinstance(obj, tuple):
            return list(obj)
        return obj

    return {
        "prompt": report.prompt,
        "overall_verdict": report.overall_verdict,
        "num_responses": len(report.responses),
        "consistency": _serialize(report.consistency) if report.consistency else None,
        "hallucination": [_serialize(h) for h in report.hallucination],
        "confidence": [_serialize(c) for c in report.confidence],
    }


def save_json_report(report: ValidationReport, output_path: str):
    data = _result_to_dict(report)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    print(_col(f"\n  💾  JSON report saved → {output_path}", Fore.GREEN))
