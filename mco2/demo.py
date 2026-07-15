"""
demo.py

Interactive CLI for testing PinoyBot language identification.
Lets you type Filipino-English passages and see per-word language tags.

Usage:
    python demo.py
"""

import re
from pinoybot import tag_language

# Tag label colors for terminal output (ANSI escape codes)
_TAG_COLORS = {
    'FIL': '\033[94m',
    'ENG': '\033[92m',
    'CS':  '\033[93m',
    'OTH': '\033[90m',
}
_RESET = '\033[0m'
_BOLD = '\033[1m'


def tokenize(text):
    """
    Split raw text into tokens, separating punctuation from words.

    Matches the dataset's tokenization where punctuation marks are
    standalone tokens (e.g., "Park." becomes ["Park", "."]).
    """
    return re.findall(r"[A-Za-z0-9\-']+|[^\s]", text)


def colorize_tag(tag):
    """Apply ANSI color to a tag label for terminal readability."""
    color = _TAG_COLORS.get(tag, '')
    return f"{color}{tag}{_RESET}"


def print_results(tokens, tags):
    """Display tagged tokens in a formatted table."""
    print(f"\n  {_BOLD}{'Word':<25} {'Tag'}{_RESET}")
    print(f"  {'-' * 35}")
    for token, tag in zip(tokens, tags):
        print(f"  {token:<25} {colorize_tag(tag)}")


def print_legend():
    """Display the tag color legend."""
    print(f"\n  {_BOLD}Tag Legend:{_RESET}")
    print(f"  {_TAG_COLORS['FIL']}FIL{_RESET} = Filipino    "
          f"{_TAG_COLORS['ENG']}ENG{_RESET} = English    "
          f"{_TAG_COLORS['CS']}CS{_RESET}  = Code-Switched    "
          f"{_TAG_COLORS['OTH']}OTH{_RESET} = Other")


def main():
    """Run the interactive PinoyBot demo loop."""
    print()
    print(f"  {_BOLD}{'=' * 52}{_RESET}")
    print(f"  {_BOLD}  PinoyBot - Filipino Code-Switched Language Tagger{_RESET}")
    print(f"  {_BOLD}{'=' * 52}{_RESET}")
    print_legend()
    print(f"\n  Type a Filipino-English sentence to analyze.")
    print(f"  Commands: {_BOLD}'quit'{_RESET} to exit, {_BOLD}'clear'{_RESET} to clear screen.\n")

    while True:
        try:
            user_input = input(f"  {_BOLD}>{_RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ('quit', 'exit', 'q'):
            print(f"  Goodbye!")
            break

        if user_input.lower() == 'clear':
            print('\033c', end='')
            print_legend()
            print()
            continue

        tokens = tokenize(user_input)
        tags = tag_language(tokens)
        print_results(tokens, tags)
        print()


if __name__ == '__main__':
    main()
