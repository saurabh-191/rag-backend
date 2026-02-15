"""
Text helpers

Small utilities to normalize, clean, and split text for downstream processing.
These helpers are intentionally lightweight and dependency-free to keep the
ingestion and RAG flow robust.
"""
from typing import List
import re

from app.core.logging import get_logger

logger = get_logger(__name__)


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace/newlines into single spaces/newlines.

    Args:
        text: Raw text

    Returns:
        Normalized text
    """
    if text is None:
        return ""
    # Replace Windows newlines, tabs, NBSP etc.
    t = re.sub(r"\r\n|\r", "\n", text)
    t = re.sub(r"[\t\u00A0]+", " ", t)
    # Collapse multiple blank lines to two newlines
    t = re.sub(r"\n{3,}", "\n\n", t)
    # Collapse multiple spaces
    t = re.sub(r" {2,}", " ", t)
    return t.strip()


def strip_control_characters(text: str) -> str:
    """Remove non-printable/control characters from text."""
    if text is None:
        return ""
    # Remove characters in Cc (control) category except newline and tab
    return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]+", "", text)


def clean_text(text: str) -> str:
    """Run common cleanups: strip controls, normalize whitespace.

    Use this before tokenization, embedding, or chunking.
    """
    if not text:
        return ""
    t = strip_control_characters(text)
    t = normalize_whitespace(t)
    return t


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

def split_sentences(text: str) -> List[str]:
    """Split text into sentence-like units (heuristic).

    This is a simple splitter intended for prompt/context formatting — not a
    linguistically perfect sentence tokenizer.
    """
    if not text:
        return []
    text = clean_text(text)
    parts = _SENTENCE_SPLIT_RE.split(text)
    # Trim and filter
    parts = [p.strip() for p in parts if p and len(p.strip()) > 0]
    return parts


def truncate(text: str, max_chars: int) -> str:
    """Truncate text to at most `max_chars` characters, preserving words.

    If the text is longer than max_chars it will cut at the last space before
    the limit to avoid mid-word splits. If no space found, it will hard-cut.
    """
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_space = cut.rfind(" ")
    if last_space > 0:
        return cut[:last_space].rstrip()
    return cut.rstrip()
