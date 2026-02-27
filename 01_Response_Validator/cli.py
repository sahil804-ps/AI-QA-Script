"""
cli.py — Command-line entry point for AI QA Validator

Usage examples:
  python cli.py --responses samples/resp1.txt samples/resp2.txt --prompt "What is AI?"
  python cli.py --inline "AI is always 100% accurate." --single
  python cli.py --responses r1.txt r2.txt r3.txt --output result.json
"""

import argparse
import sys
import os

# Add parent directory to path so we can import from 'shared'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared.ai_qa_validator import AIQAValidator
from shared.report import print_report, save_json_report


def read_file(path: str) -> str:
    if not os.path.isfile(path):
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-qa-validator",
        description=(
            "🤖 AI Response Quality Validator\n"
            "Checks AI responses for consistency, hallucination risk, and confidence.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two response files
  python cli.py --responses samples/resp1.txt samples/resp2.txt --prompt "What is gravity?"

  # Single inline response (skip consistency)
  python cli.py --inline "AI is always right." --single

  # Save JSON report
  python cli.py --responses r1.txt r2.txt --output report.json
        """,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--responses", "-r",
        nargs="+",
        metavar="FILE",
        help="Paths to text files containing AI responses (2+ for consistency check)",
    )
    input_group.add_argument(
        "--inline", "-i",
        metavar="TEXT",
        help="Pass response text directly as a string",
    )

    parser.add_argument(
        "--prompt", "-p",
        default=None,
        metavar="TEXT",
        help="The original prompt sent to the AI (optional, for context)",
    )
    parser.add_argument(
        "--single", "-s",
        action="store_true",
        help="Force single-response mode (skip consistency check)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        metavar="FILE",
        help="Save JSON report to the specified file path",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Override consistency similarity threshold (default: 0.75)",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Load responses
    if args.inline:
        responses = [args.inline]
        skip_consistency = True
    else:
        responses = [read_file(p) for p in args.responses]
        skip_consistency = args.single or len(responses) < 2

    # Inject custom threshold if provided
    validator = AIQAValidator()
    if args.threshold is not None:
        validator.consistency_checker.threshold = args.threshold

    # Run validation
    report = validator.validate(
        responses=responses,
        prompt=args.prompt,
        skip_consistency=skip_consistency,
    )

    # Print to console
    print_report(report)

    # Optionally save JSON
    if args.output:
        save_json_report(report, args.output)

    # Exit code reflects verdict
    exit_codes = {"PASS": 0, "WARN": 1, "FAIL": 2}
    sys.exit(exit_codes.get(report.overall_verdict, 2))


if __name__ == "__main__":
    main()
