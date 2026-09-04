import re
import unicodedata


def normalize_text(text: str) -> str:
    """
    Deterministically normalize input text:
    1. Apply Unicode NFKC normalization.
    2. Replace all whitespace sequences (tabs, newlines, multiple spaces) with a single space.
    3. Strip leading and trailing whitespace.
    4. Preserve meaningful character casing (no blanket lowercasing).
    """
    if not text:
        return ""

    # Unicode NFKC normalization
    nfkc_normalized = unicodedata.normalize("NFKC", text)

    # Replace repeated whitespace characters with a single standard space and strip ends
    normalized = re.sub(r"\s+", " ", nfkc_normalized).strip()

    return normalized
