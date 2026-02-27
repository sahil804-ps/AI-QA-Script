"""
interactive.py — Terminal-based interactive runner for Multi-AI QA Validator

Usage:
  python interactive.py
  python interactive.py --ais gpt gemini
"""

import sys
import time
import threading
import argparse

from colorama import init, Fore, Style

from ai_clients import AIResponse
from ai_qa_validator import AIQAValidator
from report import save_json_report

init(autoreset=True)


# ═══════════════════════════════════════════════════════════════
# Spinner for loading animation
# ═══════════════════════════════════════════════════════════════

class Spinner:
    CHARS = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message: str):
        self.message = message
        self._running = False
        self._thread = None

    def _spin(self):
        i = 0
        while self._running:
            char = self.CHARS[i % len(self.CHARS)]
            print(f"\r  {Fore.CYAN}{char}{Style.RESET_ALL}  {self.message}", end="", flush=True)
            time.sleep(0.08)
            i += 1

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self, success: bool = True, label: str = ""):
        self._running = False
        if self._thread:
            self._thread.join()
        icon = f"{Fore.GREEN}✅" if success else f"{Fore.RED}❌"
        suffix = f"  {Fore.WHITE}{label}" if label else ""
        print(f"\r  {icon}  {self.message}{suffix}{Style.RESET_ALL}")


# ═══════════════════════════════════════════════════════════════
# Pretty print AI responses before QA
# ═══════════════════════════════════════════════════════════════

def _col(text, color):
    return f"{color}{text}{Style.RESET_ALL}"


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
# Per-AI fetch with individual spinner
# ═══════════════════════════════════════════════════════════════

def fetch_with_spinners(prompt: str, selected=None) -> list[AIResponse]:
    """Fetch responses from all AIs, showing a spinner per AI."""
    from ai_clients import ALL_CLIENTS, OpenAIClient, GeminiClient, AnthropicClient
    import concurrent.futures

    name_map = {
        "gpt": OpenAIClient, "chatgpt": OpenAIClient, "openai": OpenAIClient,
        "gemini": GeminiClient, "google": GeminiClient,
        "claude": AnthropicClient, "anthropic": AnthropicClient,
    }

    if selected:
        seen = set()
        clients = []
        for s in selected:
            cls = name_map.get(s.lower())
            if cls and cls not in seen:
                clients.append(cls())
                seen.add(cls)
    else:
        clients = [cls() for cls in ALL_CLIENTS]

    print()
    print(_col("  🚀  Fetching responses from AIs...", Fore.CYAN))
    print()

    results = []
    result_lock = threading.Lock()
    spinners: dict = {}

    # Start all spinners
    for client in clients:
        sp = Spinner(f"Calling {client.ai_name} ({client.model})...")
        spinners[client.ai_name] = sp
        sp.start()

    def call_one(client):
        if not client.is_available():
            resp = AIResponse(
                ai_name=client.ai_name, model=client.model,
                response=None, success=False,
                error="API key not set"
            )
        else:
            resp = client.get_response(prompt)

        sp = spinners[client.ai_name]
        if resp.success:
            word_count = len(resp.response.split())
            sp.stop(success=True, label=f"({word_count} words)")
        else:
            sp.stop(success=False, label=f"({resp.error[:50]})")

        with result_lock:
            results.append(resp)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(clients)) as executor:
        futures = [executor.submit(call_one, c) for c in clients]
        concurrent.futures.wait(futures)

    results.sort(key=lambda r: r.ai_name)
    return results


# ═══════════════════════════════════════════════════════════════
# Main Interactive Flow
# ═══════════════════════════════════════════════════════════════

def run_interactive(selected_ais=None, output_file=None):
    print()
    print(_col("╔══════════════════════════════════════════════════════════╗", Fore.CYAN + Style.BRIGHT))
    print(_col("║      🤖  AI QA VALIDATOR — INTERACTIVE MODE              ║", Fore.CYAN + Style.BRIGHT))
    print(_col("╚══════════════════════════════════════════════════════════╝", Fore.CYAN + Style.BRIGHT))
    print()
    print(f"  {Fore.WHITE}AIs active : {Fore.YELLOW}{', '.join(selected_ais) if selected_ais else 'GPT + Gemini + Claude'}{Style.RESET_ALL}")
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
    from report import (
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
        help="Which AIs to use: gpt, gemini, claude (default: all)",
        choices=["gpt", "chatgpt", "openai", "gemini", "google", "claude", "anthropic"],
    )
    parser.add_argument(
        "--output", "-o",
        metavar="FILE",
        help="Save JSON report to file",
    )

    args = parser.parse_args()
    run_interactive(selected_ais=args.ais, output_file=args.output)
