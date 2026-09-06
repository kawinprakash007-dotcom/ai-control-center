import re
import math
from typing import Set, List, Optional, Dict, Any

from core.models.context import ContextSource, ContextPriority


# Common stop words to exclude during deterministic tokenization
STOP_WORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "where", "why", "how", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "can", "will", "just", "should",
    "now", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "doing", "would", "could"
}


def tokenize_deterministic(text: str) -> List[str]:
    """
    Extract lowercase alphanumeric tokens, filtering out stop words and single characters.
    Deterministic, pure-python, zero-dependency tokenizer.
    """
    if not text or not isinstance(text, str):
        return []
    raw_tokens = re.findall(r"[a-zA-Z0-9_\-]+", text.lower())
    return [t for t in raw_tokens if len(t) > 1 and t not in STOP_WORDS]


def compute_token_overlap(query_tokens: List[str], text_tokens: List[str]) -> float:
    """
    Compute bounded [0.0 - 1.0] token overlap between query and text.
    Combines recall against query tokens with Jaccard coefficient.
    Supports exact token matches and prefix/substring matches for meaningful stems (len >= 4).
    """
    if not query_tokens or not text_tokens:
        return 0.0

    q_set = set(query_tokens)
    t_set = set(text_tokens)

    # Calculate matched query count with prefix/substring matching
    matched_query_count = 0.0
    for q in q_set:
        if q in t_set:
            matched_query_count += 1.0
        elif any(len(q) >= 4 and len(t) >= 4 and (q.startswith(t) or t.startswith(q) or q in t or t in q) for t in t_set):
            matched_query_count += 0.85

    recall = min(1.0, matched_query_count / len(q_set))

    # Calculate Jaccard similarity on exact intersection
    intersection = q_set.intersection(t_set)
    union = q_set.union(t_set)
    jaccard = len(intersection) / len(union) if union else 0.0

    # Weighted blend giving primary emphasis to query recall
    return min(1.0, max(0.0, 0.75 * recall + 0.25 * jaccard))


def compute_recency_score(
    item_timestamp: Optional[float],
    current_time: float,
    half_life_seconds: float = 300.0,
) -> float:
    """
    Deterministic exponential decay recency calculation.
    Returns [0.0 - 1.0] where 1.0 is immediate now.
    """
    if item_timestamp is None:
        return 0.5

    age = max(0.0, current_time - item_timestamp)
    # Exponential decay: score = 0.5^(age / half_life)
    return math.exp(-0.693147 * (age / max(1.0, half_life_seconds)))


def score_candidate_relevance(
    content: str,
    goal: str,
    task: Optional[str] = None,
    source: Optional[ContextSource] = None,
    recency_score: float = 0.5,
    confidence: float = 1.0,
    has_active_failure: bool = False,
    is_visual_turn: bool = False,
) -> float:
    """
    Deterministic multi-signal relevance scoring.
    Combines:
    1. Direct goal match
    2. Current task match
    3. Source confidence & semantic relevance
    4. Modulated recency
    """
    content_tokens = tokenize_deterministic(content)
    goal_tokens = tokenize_deterministic(goal)
    task_tokens = tokenize_deterministic(task or "")

    # 1. Goal match
    goal_match = compute_token_overlap(goal_tokens, content_tokens)

    # 2. Task match
    task_match = compute_token_overlap(task_tokens, content_tokens) if task_tokens else goal_match

    # Base lexical relevance: maximum of goal match, task match, or their combination
    base_lexical = max(
        0.55 * goal_match + 0.45 * task_match,
        goal_match,
        task_match,
    )

    # Source-specific semantic boost
    source_boost = 0.0
    if source == ContextSource.CURRENT_GOAL:
        source_boost = 1.0
    elif source == ContextSource.CURRENT_TASK:
        source_boost = 0.9
    elif has_active_failure and source in (ContextSource.RECOVERY, ContextSource.VERIFICATION):
        source_boost = 0.85
    elif is_visual_turn and source == ContextSource.VISUAL_SCENE:
        source_boost = 0.8
    elif source == ContextSource.EXECUTION_RESULT:
        source_boost = max(0.65, base_lexical)
    elif source in (ContextSource.WEB_EVIDENCE, ContextSource.KNOWLEDGE):
        source_boost = base_lexical
    else:
        source_boost = base_lexical

    # Combine lexical relevance and source boost
    combined_relevance = max(base_lexical, source_boost * 0.5 + base_lexical * 0.5)

    # Recency modulation: recency enhances relevant items; it never elevates completely irrelevant items
    recency_factor = 0.88 + 0.12 * recency_score

    # Confidence weighting: verified/confident sources receive higher trust
    confidence_factor = 0.88 + (0.12 * max(0.0, min(1.0, confidence)))

    final_score = combined_relevance * recency_factor * confidence_factor
    return round(min(1.0, max(0.05, final_score)), 4)
