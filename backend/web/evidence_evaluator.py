import re
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Optional, Sequence, Set

from core.models.web import EvidenceItem, EvidenceSet
from core.models.research import EvidenceAssessment, ResearchGap, Contradiction


class EvidenceEvaluator:
    """
    Deterministic evidence evaluator for Phase 3.1 Web Intelligence.
    Assesses individual and collective evidence items for:
    - Relevance to research objective
    - Temporal freshness
    - Heuristic source and content quality
    - Objective topic coverage & research gap detection
    - Factual and numeric contradiction detection
    """

    STOP_WORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "up", "about", "into", "over", "after",
        "what", "how", "why", "when", "where", "who", "which", "is", "are",
        "was", "were", "be", "been", "being", "have", "has", "had", "do",
        "does", "did", "please", "research", "compare", "investigate",
    }

    def extract_keywords(self, text: str) -> List[str]:
        """Extract normalized informative keywords from text."""
        tokens = re.findall(r"[a-zA-Z0-9_\-\.]+", text.lower())
        return [t for t in tokens if len(t) > 1 and t not in self.STOP_WORDS]

    def calculate_relevance(self, item: EvidenceItem, objective: str) -> float:
        """
        Calculate deterministic relevance score [0.0 - 1.0] based on
        keyword and phrase matching between objective and evidence.
        """
        obj_keywords = set(self.extract_keywords(objective))
        if not obj_keywords:
            return 0.5

        title_keywords = set(self.extract_keywords(item.title))
        content_keywords = set(self.extract_keywords(item.content))
        all_item_keywords = title_keywords.union(content_keywords)

        overlap = obj_keywords.intersection(all_item_keywords)
        overlap_ratio = len(overlap) / len(obj_keywords)

        # Title match bonus
        title_overlap = obj_keywords.intersection(title_keywords)
        title_bonus = 0.2 if len(title_overlap) >= 1 else 0.0

        # Exact substring boost
        obj_clean = objective.strip().lower()
        exact_bonus = 0.2 if (obj_clean in item.title.lower() or obj_clean in item.content.lower()) else 0.0

        score = min(1.0, (overlap_ratio * 0.7) + title_bonus + exact_bonus)
        return round(score, 3)

    def calculate_freshness(self, item: EvidenceItem) -> float:
        """
        Calculate freshness score [0.0 - 1.0] from retrieval timestamp
        and recency indicators in content.
        """
        score = 0.5

        # Check ISO timestamp in retrieved_at
        if item.retrieved_at:
            try:
                dt = datetime.fromisoformat(item.retrieved_at.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                age_days = (now - dt).total_seconds() / 86400.0
                if age_days < 7:
                    score += 0.3
                elif age_days < 30:
                    score += 0.2
                elif age_days < 365:
                    score += 0.1
            except Exception:
                pass

        # Check for modern year references (2025, 2026) in title/content
        text = f"{item.title} {item.content}"
        current_year = str(datetime.now().year)
        if current_year in text or "2026" in text or "2025" in text:
            score += 0.2

        return min(1.0, round(score, 3))

    def detect_source_signal(self, item: EvidenceItem) -> str:
        """Categorize source type from domain and path signals."""
        url_lower = (item.url or "").lower()
        domain_lower = (item.domain or "").lower()

        if any(d in domain_lower for d in ("docs.", "documentation.", "datasheets.", "wiki.", "developer.")) or \
           any(p in url_lower for p in ("/docs/", "/documentation/", "/manual/", "/api/")):
            return "documentation"

        if any(d in domain_lower for d in ("github.com", "gitlab.com", "arxiv.org", "doi.org")):
            return "academic"

        if any(d in domain_lower for d in ("official", "raspberrypi.com", "nvidia.com", "microsoft.com", "google.com", ".gov", ".edu")):
            return "official"

        if any(d in domain_lower for d in ("news", "reuters.com", "theverge.com", "techcrunch.com", "arstechnica.com", "anandtech.com")):
            return "news"

        return "general"

    def calculate_quality(self, item: EvidenceItem) -> float:
        """
        Calculate heuristic source and content quality [0.0 - 1.0].
        """
        quality = 0.2  # baseline

        # HTTPS protocol
        if item.url.lower().startswith("https://"):
            quality += 0.2

        # Non-empty, informative title
        if len(item.title.strip()) > 5:
            quality += 0.2

        # Content richness (deep fetched page vs short snippet)
        content_len = len(item.content.strip())
        if content_len > 300:
            quality += 0.2
        elif content_len > 80:
            quality += 0.1

        # Authoritative or documentation source
        signal = self.detect_source_signal(item)
        if signal in ("official", "documentation", "academic"):
            quality += 0.2
        elif signal == "news":
            quality += 0.1

        return min(1.0, round(quality, 3))

    def identify_aspects(self, objective: str) -> List[str]:
        """
        Deconstruct research objective into key conceptual topics or aspects.
        Detects comparative targets, evaluation criteria, and focus dimensions.
        """
        obj_clean = objective.strip()
        aspects: List[str] = []

        # Check for comparison patterns: "compare X and Y", "X vs Y"
        compare_match = re.search(
            r"(?:compare\s+)?(.+?)\s+(?:vs\.?|versus|and|with)\s+(.+?)(?:\s+(?:for|on|in)\s+|$)",
            obj_clean,
            re.IGNORECASE,
        )
        if compare_match:
            side_a = compare_match.group(1).strip()
            side_b = compare_match.group(2).strip()
            # Clean wrappers
            side_a = re.sub(r"^(?:please\s+)?(?:research\s+)?(?:the\s+)?", "", side_a, flags=re.IGNORECASE).strip()
            side_b = re.sub(r"[.!?]+$", "", side_b).strip()
            if side_a:
                aspects.append(side_a)
            if side_b:
                aspects.append(side_b)

        # Standard technical dimensions
        dimensions = [
            ("benchmark", r"\b(benchmarks?|performance|speed|fps|tops|throughput)\b"),
            ("specifications", r"\b(specs?|specifications?|hardware|architecture|memory|ram)\b"),
            ("power", r"\b(power|watts?|consumption|efficiency|thermal)\b"),
            ("pricing", r"\b(price|pricing|cost|availability|buy|msrp)\b"),
            ("software", r"\b(software|sdk|framework|os|support|driver|ecosystem)\b"),
        ]
        for name, pattern in dimensions:
            if re.search(pattern, obj_clean, re.IGNORECASE):
                aspects.append(name)

        if not aspects:
            # Fallback to key phrases
            aspects = [obj_clean]

        return list(dict.fromkeys(aspects))

    def calculate_coverage(
        self,
        evidence: Sequence[EvidenceItem],
        objective: str,
    ) -> Tuple[float, List[ResearchGap]]:
        """
        Evaluate how well the accumulated evidence answers the objective's aspects.
        Emits structured ResearchGap items for missing dimensions.
        """
        aspects = self.identify_aspects(objective)
        if not aspects:
            return 1.0, []

        covered_aspects: Set[str] = set()
        gaps: List[ResearchGap] = []

        combined_text = " ".join(f"{item.title} {item.content}" for item in evidence).lower()

        for aspect in aspects:
            aspect_keywords = set(self.extract_keywords(aspect))
            if not aspect_keywords:
                continue

            matches = sum(1 for kw in aspect_keywords if kw in combined_text)
            coverage_ratio = matches / len(aspect_keywords)

            if coverage_ratio >= 0.5:
                covered_aspects.add(aspect)
            else:
                priority = 1 if len(aspects) <= 2 else 2
                gaps.append(
                    ResearchGap(
                        topic=aspect,
                        reason=f"Insufficient evidence collected regarding '{aspect}'.",
                        priority=priority,
                    )
                )

        coverage_score = round(len(covered_aspects) / len(aspects), 3) if aspects else 1.0
        return coverage_score, gaps

    def detect_contradictions(
        self,
        evidence: Sequence[EvidenceItem],
        objective: str,
    ) -> List[Contradiction]:
        """
        Detect conflicting statements across evidence items.
        Identifies opposing polarity (e.g. supports vs does not support)
        and divergent numeric metrics on matching subjects.
        Preserves all conflicting sources without discarding either.
        """
        contradictions: List[Contradiction] = []
        if len(evidence) < 2:
            return contradictions

        # Polarity conflict patterns
        polarity_pairs = [
            (r"\bsupports?\b", r"\b(?:does\s+not\s+support|no\s+support\s+for|cannot\s+support|unsupported)\b"),
            (r"\bcompatible\b", r"\bincompatible\b"),
            (r"\bavailable\b", r"\b(?:unavailable|discontinued|out\s+of\s+stock)\b"),
            (r"\bincluded\b", r"\b(?:not\s+included|sold\s+separately)\b"),
        ]

        # Numeric metric conflict patterns (e.g. "X TOPS" vs "Y TOPS")
        metric_pattern = r"\b(\d+(?:\.\d+)?)\s*(tops|watts?|w|fps|msrp|\$|requests per second|req/s|rps|ms)\b"

        # Pairwise comparison
        for i in range(len(evidence)):
            for j in range(i + 1, len(evidence)):
                item_a = evidence[i]
                item_b = evidence[j]

                text_a = f"{item_a.title} {item_a.content}".lower()
                text_b = f"{item_b.title} {item_b.content}".lower()

                # Check polarity pairs
                for pos_pat, neg_pat in polarity_pairs:
                    neg_in_a = bool(re.search(neg_pat, text_a))
                    clean_text_a = re.sub(neg_pat, "", text_a) if neg_in_a else text_a
                    pos_in_a = bool(re.search(pos_pat, clean_text_a))

                    neg_in_b = bool(re.search(neg_pat, text_b))
                    clean_text_b = re.sub(neg_pat, "", text_b) if neg_in_b else text_b
                    pos_in_b = bool(re.search(pos_pat, clean_text_b))

                    if (pos_in_a and not neg_in_a and neg_in_b and not pos_in_b) or \
                       (neg_in_a and not pos_in_a and pos_in_b and not neg_in_b):
                        topic_match = item_a.title or objective
                        contradictions.append(
                            Contradiction(
                                topic=topic_match,
                                evidence_ids=(item_a.id, item_b.id),
                                conflicting_claims=(
                                    f"[{item_a.title}]: Asserts positive capability/support.",
                                    f"[{item_b.title}]: Asserts negative/unsupported status.",
                                ),
                                resolved=False,
                            )
                        )
                        break

                # Check metric divergence on overlapping subjects
                metrics_a = re.findall(metric_pattern, text_a)
                metrics_b = re.findall(metric_pattern, text_b)

                unit_vals_a: Dict[str, Set[float]] = {}
                for val_str, unit in metrics_a:
                    unit_vals_a.setdefault(unit, set()).add(float(val_str))

                unit_vals_b: Dict[str, Set[float]] = {}
                for val_str, unit in metrics_b:
                    unit_vals_b.setdefault(unit, set()).add(float(val_str))

                common_units = set(unit_vals_a.keys()).intersection(unit_vals_b.keys())
                for unit in common_units:
                    vals_a = unit_vals_a[unit]
                    vals_b = unit_vals_b[unit]
                    # If mutually disjoint sets and max divergence > 30%
                    if not vals_a.intersection(vals_b):
                        max_a, min_a = max(vals_a), min(vals_a)
                        max_b, min_b = max(vals_b), min(vals_b)
                        if abs(max_a - max_b) / max(max_a, max_b, 1.0) > 0.3:
                            contradictions.append(
                                Contradiction(
                                    topic=f"Discrepancy in {unit.upper()} metrics",
                                    evidence_ids=(item_a.id, item_b.id),
                                    conflicting_claims=(
                                        f"[{item_a.title}]: Reports {max_a} {unit}.",
                                        f"[{item_b.title}]: Reports {max_b} {unit}.",
                                    ),
                                    resolved=False,
                                )
                            )

        # Deduplicate contradictions by topic
        unique_contradictions: List[Contradiction] = []
        seen_topics: Set[str] = set()
        for c in contradictions:
            if c.topic not in seen_topics:
                seen_topics.add(c.topic)
                unique_contradictions.append(c)

        return unique_contradictions

    def evaluate(
        self,
        evidence: Sequence[EvidenceItem],
        objective: str,
    ) -> Tuple[Tuple[EvidenceAssessment, ...], Tuple[ResearchGap, ...], Tuple[Contradiction, ...]]:
        """
        Execute full evaluation cycle over evidence set:
        1. Item-level assessment (relevance, freshness, quality, source signal)
        2. Set-level coverage & research gap detection
        3. Pairwise contradiction detection
        """
        contradictions = self.detect_contradictions(evidence, objective)
        contradicting_ids = {eid for c in contradictions for eid in c.evidence_ids}

        coverage_score, gaps = self.calculate_coverage(evidence, objective)

        assessments: List[EvidenceAssessment] = []
        for item in evidence:
            rel = self.calculate_relevance(item, objective)
            fresh = self.calculate_freshness(item)
            qual = self.calculate_quality(item)
            signal = self.detect_source_signal(item)
            has_contra = item.id in contradicting_ids

            assessments.append(
                EvidenceAssessment(
                    evidence_id=item.id,
                    relevance_score=rel,
                    freshness_score=fresh,
                    coverage_score=coverage_score,
                    quality_score=qual,
                    has_contradiction=has_contra,
                    source_signal=signal,
                    notes=f"Source: {signal}, rel: {rel}, qual: {qual}",
                )
            )

        return tuple(assessments), tuple(gaps), tuple(contradictions)
