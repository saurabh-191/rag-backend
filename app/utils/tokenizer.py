"""
Tokenizer utilities

Lightweight token counting helpers. Precise token counts depend on the
model/tokenizer (e.g., tiktoken). These helpers provide a reasonable
approximation and small utilities used by the pipeline for budgeting/truncation.
"""
import re
from typing import List

from app.core.logging import get_logger

logger = get_logger(__name__)

# Simple regex-based tokenization (words + punctuation)
_TOKEN_RE = re.compile(r"\w+|[^	\w\s]", re.UNICODE)


def tokens_from_text(text: str) -> List[str]:
    """Return list of token-like strings from text (heuristic)."""
    if not text:
        return []
    return _TOKEN_RE.findall(text)


def count_tokens(text: str) -> int:
    """Estimate number of tokens for a piece of text.

    This uses a fast heuristic based on word/punctuation splitting. For exact
    model token counts, integrate `tiktoken` or your model's tokenizer.
    """
    tokens = tokens_from_text(text)
    return len(tokens)


def estimate_tokens_by_chars(text: str) -> int:
    """Fallback heuristic: estimate tokens by character length (approx 4 chars/token).

    Use when a quicker estimate is needed or tokenizer isn't available.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def truncate_to_token_count(text: str, max_tokens: int) -> str:
    """Truncate text so the heuristic token count <= max_tokens.

    Strategy: keep words until token budget exhausted. This is O(n) and
    returns a substring that ends at a word boundary.
    """
    if max_tokens <= 0:
        return ""
    tokens = tokens_from_text(text)
    if len(tokens) <= max_tokens:
        return text
    # Reconstruct truncated text from tokens (simple join may lose spacing)
    # Safer approach: iterate words from original text
    words = re.split(r"(\s+)", text)
    out = []
    token_count = 0
    for part in words:
        if part.isspace():
            out.append(part)
            continue
        # Count tokens in this part
        part_tokens = tokens_from_text(part)
        if token_count + len(part_tokens) > max_tokens:
            break
        out.append(part)
        token_count += len(part_tokens)
    truncated = "".join(out).strip()
    logger.debug(f"Truncated text to approx {token_count} tokens (max {max_tokens})")
    return truncated
