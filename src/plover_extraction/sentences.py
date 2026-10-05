"""Turn raw documents into stable, addressable sentences."""

import re

from .models import Sentence


def split_sentences(raw_text: str, pattern: str) -> list[Sentence]:
    """Split raw text with a configurable regex and assign ``s1``, ``s2``, ... IDs."""
    if not raw_text or not raw_text.strip():
        raise ValueError("raw_text must be a non-empty string")
    try:
        parts = re.split(pattern, raw_text.strip())
    except re.error as error:
        raise ValueError(f"Invalid sentence_split_pattern: {error}") from error
    cleaned = [part.strip() for part in parts if part and len(part.strip().split()) > 3]
    indexed_sentences = []
    for index, text in enumerate(cleaned, 1):
        punctuation = text[-1] if text[-1] in ".!?" else "."
        content = text[:-1].rstrip() if text[-1] in ".!?" else text

        indexed_sentences.append(
            f"{content} [s{index}]{punctuation}"
        )
    return [Sentence(f"s{index}", text) for index, text in enumerate(cleaned, 1)], " ".join(indexed_sentences)
