from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union, Tuple


class AgentActionType(str, Enum):
    """
    Allowed actions in Phase 3.0 Autonomous Research Agent runtime.
    Strictly bounded; no arbitrary tool execution or system access allowed.
    """
    SEARCH = "search"
    FETCH = "fetch"
    FINISH = "finish"


@dataclass(frozen=True)
class AgentAction:
    """
    Structured action proposed by a reasoning model or provider.
    Subject to strict runtime validation and bounds before execution.

    Attributes:
        action_type: Action to perform ('search', 'fetch', 'finish').
        parameters: Action-specific arguments (e.g. query, url).
        reason: Justification or reasoning behind selecting this action.
        confidence: Optional confidence score [0.0 - 1.0].
    """
    action_type: Union[AgentActionType, str]
    parameters: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        atype = self.action_type.value if isinstance(self.action_type, AgentActionType) else str(self.action_type)
        return {
            "action_type": atype,
            "parameters": dict(self.parameters),
            "reason": self.reason,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentAction":
        raw_type = data.get("action_type", "")
        try:
            action_type = AgentActionType(str(raw_type).strip().lower())
        except (ValueError, KeyError):
            action_type = raw_type
        return cls(
            action_type=action_type,
            parameters=dict(data.get("parameters", {})),
            reason=str(data.get("reason", "")),
            confidence=float(data.get("confidence", 1.0)),
        )


@dataclass(frozen=True)
class ResearchObservation:
    """
    Bounded observation returned to the reasoning provider after an action executes.
    Strictly bounded to prevent prompt explosion and context poisoning.

    Attributes:
        action_type: The action that produced this observation.
        success: Whether the action succeeded at runtime.
        summary: Short human-readable summary of the outcome.
        data: Bounded structured data (e.g. snippets, titles, status codes).
        error: Optional error description if the action failed.
    """
    action_type: str
    success: bool
    summary: str
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type,
            "success": self.success,
            "summary": self.summary,
            "data": dict(self.data),
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchObservation":
        return cls(
            action_type=str(data.get("action_type", "")),
            success=bool(data.get("success", False)),
            summary=str(data.get("summary", "")),
            data=dict(data.get("data", {})),
            error=data.get("error"),
        )


@dataclass(frozen=True)
class ResearchLimits:
    """
    Runtime-owned resource and execution limits for the autonomous research loop.
    Guarantees strict termination and prevents infinite loops or runaway execution.

    The model can NEVER alter these limits.
    """
    max_iterations: int = 10
    max_searches: int = 10
    max_fetches: int = 5
    max_evidence: int = 50
    min_evidence: int = 1
    max_invalid_actions: int = 3
    timeout_seconds: float = 10.0

    def enforce_caps(self) -> "ResearchLimits":
        """Defensively cap all limits to hard runtime maxima."""
        return ResearchLimits(
            max_iterations=max(1, min(self.max_iterations, 10)),
            max_searches=max(1, min(self.max_searches, 10)),
            max_fetches=max(0, min(self.max_fetches, 5)),
            max_evidence=max(1, min(self.max_evidence, 50)),
            min_evidence=max(1, self.min_evidence),
            max_invalid_actions=max(1, min(self.max_invalid_actions, 5)),
            timeout_seconds=max(0.1, min(self.timeout_seconds, 60.0)),
        )


@dataclass(frozen=True)
class EvidenceAssessment:
    """
    Structured, deterministic evaluation signals for a collected EvidenceItem.
    Provides clear supporting signals for reasoning without claiming absolute truth.

    Attributes:
        evidence_id: Identifier of evaluated EvidenceItem.
        relevance_score: Relevancy to research objective [0.0 - 1.0].
        freshness_score: Timeliness and temporal freshness [0.0 - 1.0].
        coverage_score: Breadth of information provided [0.0 - 1.0].
        quality_score: Heuristic source/content quality [0.0 - 1.0].
        has_contradiction: Whether this evidence conflicts with other items.
        source_signal: Category of source (e.g. 'official', 'documentation', 'general').
        notes: Supporting summary of evaluation findings.
    """
    evidence_id: str
    relevance_score: float = 0.5
    freshness_score: float = 0.5
    coverage_score: float = 0.5
    quality_score: float = 0.5
    has_contradiction: bool = False
    source_signal: str = "general"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "relevance_score": self.relevance_score,
            "freshness_score": self.freshness_score,
            "coverage_score": self.coverage_score,
            "quality_score": self.quality_score,
            "has_contradiction": self.has_contradiction,
            "source_signal": self.source_signal,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceAssessment":
        return cls(
            evidence_id=str(data.get("evidence_id", "")),
            relevance_score=float(data.get("relevance_score", 0.5)),
            freshness_score=float(data.get("freshness_score", 0.5)),
            coverage_score=float(data.get("coverage_score", 0.5)),
            quality_score=float(data.get("quality_score", 0.5)),
            has_contradiction=bool(data.get("has_contradiction", False)),
            source_signal=str(data.get("source_signal", "general")),
            notes=str(data.get("notes", "")),
        )


@dataclass(frozen=True)
class ResearchGap:
    """
    Identified deficiency or unanswered facet in the current evidence set.
    Drives adaptive query generation and follow-up fetches.

    Attributes:
        topic: Specific missing aspect or query focus.
        reason: Explanation of why this information is needed.
        priority: Urgency of gap (1=high, 2=medium, 3=low).
    """
    topic: str
    reason: str
    priority: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "reason": self.reason,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchGap":
        return cls(
            topic=str(data.get("topic", "")),
            reason=str(data.get("reason", "")),
            priority=int(data.get("priority", 1)),
        )


@dataclass(frozen=True)
class Contradiction:
    """
    Detected factual or numeric discrepancy between two or more evidence items.
    Preserves all conflicting sources without prematurely discarding either.

    Attributes:
        topic: Subject or parameter of disagreement.
        evidence_ids: Identifiers of the contradictory EvidenceItems.
        conflicting_claims: Excerpts or summaries of opposing claims.
        resolved: Whether the conflict has been clarified by further evidence.
    """
    topic: str
    evidence_ids: Tuple[str, ...]
    conflicting_claims: Tuple[str, ...]
    resolved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "evidence_ids": list(self.evidence_ids),
            "conflicting_claims": list(self.conflicting_claims),
            "resolved": self.resolved,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Contradiction":
        return cls(
            topic=str(data.get("topic", "")),
            evidence_ids=tuple(data.get("evidence_ids", ())),
            conflicting_claims=tuple(data.get("conflicting_claims", ())),
            resolved=bool(data.get("resolved", False)),
        )
