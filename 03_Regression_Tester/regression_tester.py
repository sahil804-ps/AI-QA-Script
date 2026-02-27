"""
regression_tester.py — Prompt Regression Tester (Day 3-4 of AI-QA-Toolkit)

Checks how different models (GPT, Gemini, Claude) respond to the SAME prompt
and highlights the semantic differences. Supports BATCH MODE.
"""

import sys
import os
import argparse
import json
from datetime import datetime

# Add parent directory to path so we can import from 'shared'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared.ai_clients import fetch_with_spinners
from shared.ai_qa_validator import AIQAValidator
from shared.report import save_json_report, _col
from colorama import Fore, Style, init

init(autoreset=True)

def print_regression_header(prompt: str, current: int = None, total: int = None):
    batch_info = f" [{current}/{total}]" if current and total else ""
    print()
    print(_col("╔══════════════════════════════════════════════════════════╗", Fore.MAGENTA + Style.BRIGHT))
    print(_col(f"║          🧪  PROMPT REGRESSION TESTER{batch_info:<11}      ║", Fore.MAGENTA + Style.BRIGHT))
    print(_col("╚══════════════════════════════════════════════════════════╝", Fore.MAGENTA + Style.BRIGHT))
    print()
    print(f"  {Fore.WHITE}Prompt: {Fore.YELLOW}\"{prompt}\"{Style.RESET_ALL}")
    print()

def run_regression_test(prompt: str, selected_ais=None, output=None, current=None, total=None):
    print_regression_header(prompt, current, total)
    
    # 1. Fetch from all AIs
    responses = fetch_with_spinners(prompt, selected=selected_ais)
    
    # 2. Filter success
    successful = [r for r in responses if r.success and r.response]
    
    if len(successful) < 2:
        print(f"\n  {Fore.RED}❌ Error: Regression test requires at least 2 successful AI responses.{Style.RESET_ALL}")
        return None

    # 3. Use Validator for Analysis
    validator = AIQAValidator()
    report = validator.validate(
        responses=[r.response for r in successful],
        prompt=prompt,
        skip_consistency=False
    )

    # 4. Custom Regression Output (Comparing Models)
    print()
    print(_col("═══ 🔬 MODEL COMPARISON (REGRESSION) ════════════════════════", Fore.MAGENTA))
    print()
    
    print(f"  {'Model':<15} {'Words':<10} {'Quality':<12} {'Verdict'}")
    print(f"  {'─'*15} {'─'*10} {'─'*12} {'─'*10}")
    
    for i, r in enumerate(successful):
        # Match model by name or check index
        conf = report.confidence[i]
        hall = report.hallucination[i]
        
        quality_icon = "🌟" if conf.score >= 85 else "👍" if conf.score >= 70 else "😐"
        risk_icon = "✅" if hall.risk_level == "LOW" else "⚠️ " if hall.risk_level == "MEDIUM" else "🚨"
        
        print(f"  {r.ai_name:<15} {conf.word_count:<10} {quality_icon} {conf.score:<8.1f} {risk_icon} {hall.risk_level}")

    # 5. Consistency/Difference Insight
    print()
    if report.consistency.verdict == "CONSISTENT":
        print(f"  {Fore.GREEN}✅ Models are semantically CONSISTENT ({report.consistency.average_similarity*100:.1f}% match).")
    else:
        print(f"  {Fore.RED}❌ Models are INCONSISTENT ({report.consistency.average_similarity*100:.1f}% similarity).")

    # 6. Save Report
    if output:
        save_json_report(report, output)
    
    return report

def main():
    parser = argparse.ArgumentParser(description="AI Prompt Regression Tester (Batch Automated)")
    parser.add_argument("prompt", nargs="?", help="Specific prompt to test (optional if using --batch)")
    parser.add_argument("--batch", "-b", metavar="FILE", help="JSON file containing a list of prompts")
    parser.add_argument(
        "--ais", nargs="+",
        metavar="AI",
        help="Which AIs to use: gpt, gemini, claude, openrouter, llama, deepseek (default: all)",
        choices=["gpt", "chatgpt", "openai", "gemini", "google", "claude", "anthropic", "openrouter", "or", "llama", "deepseek", "ds"],
    )
    parser.add_argument("--output", "-o", help="Save regression report to JSON (or directory if batch)")
    
    args = parser.parse_args()

    # Determine prompts
    prompts = []
    if args.batch:
        if not os.path.exists(args.batch):
            print(f"{Fore.RED}Error: Batch file {args.batch} not found.{Style.RESET_ALL}")
            sys.exit(1)
        with open(args.batch, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                prompts = data if isinstance(data, list) else [data]
            except Exception:
                # Try reading line by line if not JSON
                f.seek(0)
                prompts = [line.strip() for line in f if line.strip()]
    elif args.prompt:
        prompts = [args.prompt]
    else:
        print(f"{Fore.YELLOW}Usage: Provide a prompt or use --batch <file>{Style.RESET_ALL}")
        sys.exit(1)

    # Output directory for batch
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output or (f"reports/batch_{timestamp}" if len(prompts) > 1 else None)

    results = []
    total = len(prompts)
    
    for i, p in enumerate(prompts, 1):
        out_file = None
        if out_dir:
            if total > 1:
                out_file = os.path.join(out_dir, f"report_{i:02d}.json")
            else:
                out_file = out_dir if out_dir.endswith(".json") else os.path.join(out_dir, "report.json")
        
        rep = run_regression_test(p, selected_ais=args.ais, output=out_file, current=i, total=total)
        if rep:
            results.append(rep)

    if total > 1:
        print()
        print(_col("╔══════════════════════════════════════════════════════════╗", Fore.GREEN + Style.BRIGHT))
        print(_col("║          🏁  BATCH AUTOMATION COMPLETE                  ║", Fore.GREEN + Style.BRIGHT))
        print(_col("╚══════════════════════════════════════════════════════════╝", Fore.GREEN + Style.BRIGHT))
        print(f"  Processed {len(results)}/{total} prompts successfully.")
        if out_dir:
            print(f"  All reports saved in: {Fore.YELLOW}{out_dir}")
        print()

if __name__ == "__main__":
    main()
