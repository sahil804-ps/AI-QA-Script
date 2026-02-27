"""
interactive.py — Terminal-based interactive runner for Multi-AI QA Validator

Usage:
  python interactive.py
  python interactive.py --ais gpt gemini
"""

import sys
import argparse
import os

from colorama import init, Fore, Style

# Add parent directory to path so we can import from 'shared'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared.ai_clients import AIResponse, Spinner, fetch_with_spinners
from shared.ai_qa_validator import AIQAValidator
from shared.report import save_json_report, _col

init(autoreset=True)


# ═══════════════════════════════════════════════════════════════
# Pretty print AI responses before QA
# ═══════════════════════════════════════════════════════════════

def print_ai_responses(responses: list[AIResponse]):
    print()
    print(_col("═" * 60, Fore.CYAN))
    print(_col("  📥  AI RESPONSES RECEIVED", Fore.CYAN + Style.BRIGHT))
    print(_col("═" * 60, Fore.CYAN))
    for r in responses:
        if r.success and r.response:
            print()
            print(_col(f"  🤖  {r.ai_name}  [{r.model}]", Fore.YELLOW + Style.BRIGHT))
            print(_col("  " + "─" * 56, Fore.YELLOW))
            # Print first 300 chars of response
            preview = r.response[:300]
            if len(r.response) > 300:
                preview += f"\n  ... [{len(r.response) - 300} more chars]"
            for line in preview.splitlines():
                print(f"  {Fore.WHITE}{line}{Style.RESET_ALL}")
        else:
            print()
            print(_col(f"  ❌  {r.ai_name}  [{r.model}]  —  {r.error}", Fore.RED))
    print()


# ═══════════════════════════════════════════════════════════════
# Main Interactive Flow
# ═══════════════════════════════════════════════════════════════

def run_interactive(selected_ais=None, output_file=None):
    print()
    print(_col("╔══════════════════════════════════════════════════════════╗", Fore.CYAN + Style.BRIGHT))
    print(_col("║      🤖  AI QA VALIDATOR — INTERACTIVE MODE              ║", Fore.CYAN + Style.BRIGHT))
    print(_col("╚══════════════════════════════════════════════════════════╝", Fore.CYAN + Style.BRIGHT))
    print()
    active_ais = selected_ais if selected_ais else ["GPT", "Gemini", "Claude", "OpenRouter"]
    print(f"  {Fore.WHITE}AIs active : {Fore.YELLOW}{' + '.join(active_ais)}{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}Type your prompt and press Enter.{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}Type {Fore.YELLOW}exit{Fore.WHITE} to quit.{Style.RESET_ALL}")
    print()

    while True:
        try:
            prompt = input(_col("  📝  Your Prompt: ", Fore.GREEN + Style.BRIGHT)).strip()
        except (KeyboardInterrupt, EOFError):
            print()
            print(_col("\n  👋  Bye!", Fore.YELLOW))
            sys.exit(0)

        if not prompt:
            print(_col("  ⚠️   Prompt empty hai — kuch type karo!", Fore.YELLOW))
            continue

        if prompt.lower() in ("exit", "quit", "q"):
            print(_col("\n  👋  Bye!", Fore.YELLOW))
            break

        # Fetch from all AIs
        ai_responses = fetch_with_spinners(prompt, selected=selected_ais)

        # Show raw responses
        print_ai_responses(ai_responses)

        # Filter successful responses for QA
        successful = [r for r in ai_responses if r.success and r.response]

        if not successful:
            print(_col("  ❌  Koi bhi AI response nahi de paaya. API keys check karo .env mein.", Fore.RED))
            print()
            continue

        if len(successful) == 1:
            print(_col(f"  ⚠️   Sirf {successful[0].ai_name} ka response mila — consistency check skip hogi.", Fore.YELLOW))

        # Run QA
        validator = AIQAValidator()
        responses_text = [r.response for r in successful]
        labels = [f"{r.ai_name}" for r in successful]

        report = validator.validate(
            responses=responses_text,
            prompt=prompt,
            skip_consistency=(len(responses_text) < 2),
        )

        # Override labels in report output for named AIs
        _print_report_with_labels(report, labels)

        # Save JSON if needed
        if output_file:
            save_json_report(report, output_file)

        print()
        again = input(_col("  🔄  Ek aur prompt test karna hai? (y/n): ", Fore.CYAN)).strip().lower()
        if again not in ("y", "yes", "ha", "haan", "h"):
            print(_col("\n  👋  Done! Bye!", Fore.YELLOW))
            break
        print()


def _print_report_with_labels(report, labels):
    """Print report using AI names as labels instead of generic 'Response 1'."""
    from shared.report import (
        print_consistency, print_hallucination,
        print_confidence, print_overall
    )
    from colorama import Fore, Style

    print()
    print(_col("╔══════════════════════════════════════════════════════════╗", Fore.CYAN + Style.BRIGHT))
    print(_col("║          📊  QA VALIDATION REPORT                        ║", Fore.CYAN + Style.BRIGHT))
    print(_col("╚══════════════════════════════════════════════════════════╝", Fore.CYAN + Style.BRIGHT))
    print()

    label_tags = [f"[{name}]" for name in labels]
    print_consistency(report.consistency)
    print_hallucination(report.hallucination, label_tags)
    print_confidence(report.confidence, label_tags)
    print_overall(report.overall_verdict, report.prompt)



# ═══════════════════════════════════════════════════════════════
# CLI Entry
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="🤖 AI QA Validator — Interactive Multi-AI Mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python interactive.py
  python interactive.py --ais gpt gemini
  python interactive.py --ais gemini --output report.json
        """
    )
    parser.add_argument(
        "--ais", nargs="+",
        metavar="AI",
        help="Which AIs to use: gpt, gemini, claude, openrouter, llama, deepseek (default: all)",
        choices=["gpt", "chatgpt", "openai", "gemini", "google", "claude", "anthropic", "openrouter", "or", "llama", "deepseek", "ds"],
    )
    parser.add_argument(
        "--output", "-o",
        metavar="FILE",
        help="Save JSON report to file",
    )

    args = parser.parse_args()
    run_interactive(selected_ais=args.ais, output_file=args.output)
