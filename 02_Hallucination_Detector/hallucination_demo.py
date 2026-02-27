"""
hallucination_demo.py — Hallucination Detector (Day 2 of AI-QA-Toolkit)

Standalone module to test responses against hallucination patterns.
"""

import sys
import os
import argparse

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared.ai_qa_validator import HallucinationDetector
from shared.report import _col, _verdict_color, _badge
from colorama import Fore, Style, init

init(autoreset=True)

def demo_hallucination(text: str):
    detector = HallucinationDetector()
    result = detector.detect(text)
    
    print()
    print(_col("╔══════════════════════════════════════════════════════════╗", Fore.RED + Style.BRIGHT))
    print(_col("║          🚨  HALLUCINATION SCANNER                      ║", Fore.RED + Style.BRIGHT))
    print(_col("╚══════════════════════════════════════════════════════════╝", Fore.RED + Style.BRIGHT))
    print()
    
    vc = _verdict_color(result.risk_level)
    print(f"  Risk Level     : {_col(f'{_badge(result.risk_level)} {result.risk_level}', vc + Style.BRIGHT)}")
    print(f"  Risk Score     : {result.risk_score}")
    print(f"  Sentences      : {result.total_sentences}")
    print(f"  Flagged        : {result.flagged_sentences}")
    print()
    
    _col_map = {
        "overconfident": Fore.RED,
        "suspicious_numeric": Fore.MAGENTA,
        "hedged": Fore.YELLOW
    }

    if result.matches:
        print(f"  {Fore.WHITE}Flagged Items:{Style.RESET_ALL}")
        for m in result.matches:
            c = _col_map.get(m.category, Fore.WHITE)
            print(f"    ↳ [{_col(m.category.upper(), c)}] \"{m.matched_text}\"")
            print(f"      {Fore.BLACK}{Style.BRIGHT}Context: {m.sentence[:100]}...{Style.RESET_ALL}")
    else:
        print(f"  {Fore.GREEN}✅ No hallucination patterns found in this text.{Style.RESET_ALL}")
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Hallucination Scanner")
    parser.add_argument("--text", "-t", help="Text to scan for hallucinations")
    parser.add_argument("--file", "-f", help="File to scan")
    
    args = parser.parse_args()
    
    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            content = f.read()
    elif args.text:
        content = args.text
    else:
        print("Please provide --text or --file")
        sys.exit(1)
        
    demo_hallucination(content)
