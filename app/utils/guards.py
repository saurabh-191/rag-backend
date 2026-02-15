"""
Prompt & query guards

Basic heuristics to detect malicious or dangerous user inputs and prompt
injection patterns. These are not foolproof; treat them as a first line of
defense and combine with runtime/LLM-level guardrails.
"""
import re
from typing import Tuple

from app.core.logging import get_logger

logger = get_logger(__name__)

# Simple blacklist of dangerous tokens/commands
_DANGEROUS_PATTERNS = [
    r"rm\s+-rf",
    r"sudo\s+",
    r"\bshutdown\b",
    r"\bpoweroff\b",
    r"\breboot\b",
    r"\bcron\b",
    r"\bdelete\b.*\b(database|table|file|files)\b",
    r"\bdrop\b.*\b(table|database)\b",
    r"import\s+subprocess",
    r"os\.system\(",
    r"exec\(",
    r"eval\(",
    r"open\(.*\)",
    r"curl\s+http",
    r"wget\s+http",
]

# Heuristics for prompt-injection indicators
_PROMPT_INJECTION_PATTERNS = [
    r"ignore (?:previous|above|all) instructions",
    r"disregard (?:previous|above|all) instructions",
    r"forget (?:the )?previous",
    r"not follow the earlier",
    r"respond with.*\bflag\b",
    r"output only the following",
    r"do not include any citations",
]

_COMPILED_DANGEROUS = [re.compile(p, re.IGNORECASE) for p in _DANGEROUS_PATTERNS]
_COMPILED_INJECTION = [re.compile(p, re.IGNORECASE) for p in _PROMPT_INJECTION_PATTERNS]


def is_safe_query(query: str) -> bool:
    """Quick check for obviously unsafe queries.

    Returns True if query passes the simple blacklist checks, False otherwise.
    """
    if not query or not query.strip():
        return False
    for patt in _COMPILED_DANGEROUS:
        if patt.search(query):
            logger.warning(f"Dangerous pattern matched in query: {patt.pattern}")
            return False
    return True


def detect_prompt_injection(query: str) -> Tuple[bool, str]:
    """Detect likely prompt injection patterns.

    Returns (is_safe, message). If is_safe is False, message explains why.
    """
    if not query:
        return True, ""
    for patt in _COMPILED_INJECTION:
        m = patt.search(query)
        if m:
            msg = f"Prompt-injection pattern: {m.group(0)}"
            logger.warning(msg)
            return False, msg
    return True, ""


def sanitize_input(query: str) -> str:
    """Basic sanitizer: removes null bytes and control characters, trims.

    This should be used only as a convenience; do not rely on sanitization
    alone for security-critical checks.
    """
    if not query:
        return ""
    # Remove NUL and other control characters except newline/tab
    sanitized = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]+", "", query)
    return sanitized.strip()
