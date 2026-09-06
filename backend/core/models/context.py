from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Tuple, Set

from core.models.verification import VerificationResult
from core.models.recovery import RecoveryContext
from core.models.perception import VisualScene, GroundedTarget


@dataclass
class Context:
    """
    Legacy context model preserved for backward compatibility.
    """
    current_app: str | None = None
    current_goal: str | None = None
    last_tool: str | None = None
    last_user_message: str | None = None
    last_ai_response: str | None = None
    conversation_count: int = 0


class ContextSource(str, Enum):
    """
    Explicit provenance sources for cognitive context items.
    """
    CURRENT_GOAL = "current_goal"
    CURRENT_TASK = "current_task"
    RECENT_CONVERSATION = "recent_conversation"
    MEMORY = "memory"
    KNOWLEDGE = "knowledge"
    WEB_EVIDENCE = "web_evidence"
    RESEARCH = "research"
    VISUAL_SCENE = "visual_scene"
    EXECUTION_RESULT = "execution_result"
    VERIFICATION = "verification"
    RECOVERY = "recovery"
    CAPABILITY_STATE = "capability_state"
    WORLD_STATE = "world_state"


class ContextPriority(IntEnum):
    """
    Bounded priority hierarchy for context selection.
    Higher values take precedence during budget allocation.
    """
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    HIGHEST = 4


class SensitivityLevel(str, Enum):
    """
    Data privacy and confidentiality tiers for context items.
    """
    PUBLIC = "public"
    INTERNAL = "internal"
    SENSITIVE = "sensitive"
    SECRET = "secret"


class AttentionFocus(str, Enum):
    """
    Small, explicit vocabulary of cognitive focus areas that influence
    deterministic context ranking and prioritization.
    """
    GOAL = "goal"
    PLANNING = "planning"
    EXECUTION = "execution"
    VERIFICATION = "verification"
    RECOVERY = "recovery"
    VISUAL_TARGET = "visual_target"
    RESEARCH = "research"
    MEMORY_RECALL = "memory_recall"


@dataclass(frozen=True)
class ContextItem:
    """
    Model-neutral representation of a candidate context item for the current reasoning turn.
    CRITICAL: ContextItem is a temporary working set item, NOT persistent memory.
    """
    item_id: str
    source: ContextSource
    content: str
    relevance: float = 0.5
    priority: ContextPriority = ContextPriority.MEDIUM
    recency: float = 0.0
    confidence: float = 1.0
    timestamp: Optional[float] = None
    sensitivity: SensitivityLevel = SensitivityLevel.PUBLIC
    token_estimate: int = 0
    is_protected: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not (0.0 <= self.relevance <= 1.0):
            raise ValueError(f"ContextItem relevance must be in range [0.0, 1.0], got {self.relevance}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"ContextItem confidence must be in range [0.0, 1.0], got {self.confidence}")
        if self.token_estimate == 0 and self.content:
            # Deterministic token approximation: ~4 characters per token
            est = max(1, len(self.content) // 4)
            object.__setattr__(self, "token_estimate", est)

    def compact(self, max_chars: int) -> "ContextItem":
        """
        Deterministically truncate content to max_chars if oversized,
        preserving provenance and metadata.
        """
        if len(self.content) <= max_chars:
            return self

        truncated = self.content[:max_chars].rstrip() + " ...[truncated]"
        new_token_est = max(1, len(truncated) // 4)
        new_metadata = dict(self.metadata)
        new_metadata["original_length"] = len(self.content)
        new_metadata["compacted"] = True

        return ContextItem(
            item_id=self.item_id,
            source=self.source,
            content=truncated,
            relevance=self.relevance,
            priority=self.priority,
            recency=self.recency,
            confidence=self.confidence,
            timestamp=self.timestamp,
            sensitivity=self.sensitivity,
            token_estimate=new_token_est,
            is_protected=self.is_protected,
            metadata=new_metadata,
        )

    @property
    def is_sensitive(self) -> bool:
        return self.sensitivity in (SensitivityLevel.SENSITIVE, SensitivityLevel.SECRET)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "source": self.source.value,
            "content": self.content if not self.is_sensitive else "[REDACTED_SENSITIVE]",
            "relevance": self.relevance,
            "priority": int(self.priority),
            "recency": self.recency,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "sensitivity": self.sensitivity.value,
            "token_estimate": self.token_estimate,
            "is_protected": self.is_protected,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ContextBudget:
    """
    Deterministic context budget constraints.
    Ensures context never grows without bounds.
    """
    max_items: int = 20
    max_tokens: int = 4000
    max_content_length_per_item: int = 1000

    def __post_init__(self):
        if self.max_items <= 0:
            raise ValueError(f"max_items must be positive, got {self.max_items}")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
        if self.max_content_length_per_item <= 0:
            raise ValueError(f"max_content_length_per_item must be positive, got {self.max_content_length_per_item}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_items": self.max_items,
            "max_tokens": self.max_tokens,
            "max_content_length_per_item": self.max_content_length_per_item,
        }


@dataclass(frozen=True)
class ContextSelection:
    """
    Structured outcome of a context selection operation.
    Provides structured items, provenance, and budget audit trail.
    """
    selected_items: Tuple[ContextItem, ...] = field(default_factory=tuple)
    omitted_items: Tuple[ContextItem, ...] = field(default_factory=tuple)
    total_candidates: int = 0
    selection_reason: str = ""
    budget: ContextBudget = field(default_factory=ContextBudget)
    attention_focus: AttentionFocus = AttentionFocus.GOAL
    provenance_metadata: Dict[str, Any] = field(default_factory=dict)

    def get_items_by_source(self, source: ContextSource) -> Tuple[ContextItem, ...]:
        return tuple(item for item in self.selected_items if item.source == source)

    def has_sensitive_content(self) -> bool:
        return any(item.is_sensitive for item in self.selected_items)

    @property
    def total_tokens(self) -> int:
        return sum(item.token_estimate for item in self.selected_items)

    @property
    def item_count(self) -> int:
        return len(self.selected_items)

    def formatted_summary(self) -> str:
        """Structured text summary of selected items by source."""
        lines = [f"=== ContextSelection (Focus: {self.attention_focus.value}, Items: {self.item_count}, Tokens: ~{self.total_tokens}) ==="]
        for item in self.selected_items:
            prot = " [PROTECTED]" if item.is_protected else ""
            lines.append(f"[{item.source.value.upper()} | P:{item.priority.name} | R:{item.relevance:.2f}{prot}] {item.content}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_items": [item.to_dict() for item in self.selected_items],
            "omitted_items_count": len(self.omitted_items),
            "total_candidates": self.total_candidates,
            "selection_reason": self.selection_reason,
            "budget": self.budget.to_dict(),
            "attention_focus": self.attention_focus.value,
            "total_tokens": self.total_tokens,
            "provenance_metadata": dict(self.provenance_metadata),
        }


@dataclass(frozen=True)
class CognitiveState:
    """
    Compact, bounded model representing the current cognitive state for a reasoning step.
    Prevents passing unbounded raw database state or full history.
    """
    original_goal: str
    current_task: Optional[str] = None
    current_plan_step: Optional[str] = None
    recent_execution_outcomes: Tuple[Any, ...] = field(default_factory=tuple)
    verification_state: Optional[VerificationResult] = None
    recovery_state: Optional[RecoveryContext] = None
    visual_scene: Optional[VisualScene] = None
    grounded_candidates: Tuple[GroundedTarget, ...] = field(default_factory=tuple)
    memory_items: Tuple[Any, ...] = field(default_factory=tuple)
    knowledge_items: Tuple[Any, ...] = field(default_factory=tuple)
    web_evidence: Tuple[Any, ...] = field(default_factory=tuple)
    world_conditions: Tuple[Any, ...] = field(default_factory=tuple)
    research_state: Optional[Any] = None
    available_capabilities: Tuple[str, ...] = field(default_factory=tuple)
    attention_focus: AttentionFocus = AttentionFocus.GOAL
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.original_goal or not isinstance(self.original_goal, str) or not self.original_goal.strip():
            raise ValueError("CognitiveState original_goal must be a non-empty string.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_goal": self.original_goal,
            "current_task": self.current_task,
            "current_plan_step": self.current_plan_step,
            "recent_execution_outcomes_count": len(self.recent_execution_outcomes),
            "has_verification": self.verification_state is not None,
            "has_recovery": self.recovery_state is not None,
            "has_visual_scene": self.visual_scene is not None,
            "grounded_candidates_count": len(self.grounded_candidates),
            "memory_items_count": len(self.memory_items),
            "knowledge_items_count": len(self.knowledge_items),
            "web_evidence_count": len(self.web_evidence),
            "world_conditions_count": len(self.world_conditions),
            "has_research_state": self.research_state is not None,
            "available_capabilities": list(self.available_capabilities),
            "attention_focus": self.attention_focus.value,
            "metadata": dict(self.metadata),
        }