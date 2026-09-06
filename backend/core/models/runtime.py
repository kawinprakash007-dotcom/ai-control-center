from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Dict, List, Optional, Tuple

from core.models.request import Request
from core.models.decision import Decision, ExecutionMode
from core.models.plan import Plan
from core.models.task import Task
from core.models.result import Result
from core.models.verification import VerificationResult
from core.models.recovery import RecoveryContext
from core.models.context import ContextSelection
from core.models.reasoning import ReasoningResponse, ProposalValidationResult
from core.models.model_router import RoutingResult
from core.models.policy import PolicyResult
from core.models.pipeline import PipelineResult


class CognitiveStage(str, Enum):
    """
    Canonical lifecycle stages of an ATLAS cognitive turn.
    Represents the current phase of cognition and execution.
    """
    RECEIVED = "RECEIVED"
    UNDERSTANDING = "UNDERSTANDING"
    DECISION = "DECISION"
    PLANNING = "PLANNING"
    CONTEXT = "CONTEXT"
    ROUTING = "ROUTING"
    REASONING = "REASONING"
    PROPOSAL = "PROPOSAL"
    VALIDATION = "VALIDATION"
    POLICY = "POLICY"
    EXECUTION = "EXECUTION"
    OBSERVATION = "OBSERVATION"
    VERIFICATION = "VERIFICATION"
    RECOVERY = "RECOVERY"
    MEMORY = "MEMORY"
    RESPONSE = "RESPONSE"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"


class TurnStatus(str, Enum):
    """
    Lifecycle status of a cognitive turn.
    Separated from stage: a turn can be in POLICY stage while WAITING_FOR_USER.
    """
    RUNNING = "RUNNING"
    WAITING_FOR_USER = "WAITING_FOR_USER"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"


class CognitiveEventType(str, Enum):
    """
    Bounded vocabulary of structured domain events emitted during a cognitive turn.
    """
    TURN_STARTED = "TURN_STARTED"
    STAGE_STARTED = "STAGE_STARTED"
    STAGE_COMPLETED = "STAGE_COMPLETED"
    STAGE_FAILED = "STAGE_FAILED"
    DECISION_MADE = "DECISION_MADE"
    PLAN_CREATED = "PLAN_CREATED"
    CONTEXT_SELECTED = "CONTEXT_SELECTED"
    MODEL_ROUTED = "MODEL_ROUTED"
    REASONING_COMPLETED = "REASONING_COMPLETED"
    PROPOSAL_CREATED = "PROPOSAL_CREATED"
    PROPOSAL_VALIDATED = "PROPOSAL_VALIDATED"
    PROPOSAL_REJECTED = "PROPOSAL_REJECTED"
    POLICY_DECIDED = "POLICY_DECIDED"
    TOOL_EXECUTED = "TOOL_EXECUTED"
    OBSERVATION_RECEIVED = "OBSERVATION_RECEIVED"
    VERIFICATION_COMPLETED = "VERIFICATION_COMPLETED"
    RECOVERY_STARTED = "RECOVERY_STARTED"
    REPLAN_CREATED = "REPLAN_CREATED"
    MEMORY_UPDATED = "MEMORY_UPDATED"
    RESPONSE_COMPOSED = "RESPONSE_COMPOSED"
    TURN_COMPLETED = "TURN_COMPLETED"
    TURN_WAITING = "TURN_WAITING"
    TURN_ABORTED = "TURN_ABORTED"


# Sensitive keys that must be redacted from event metadata
_SENSITIVE_KEYS = {
    "password", "secret", "token", "auth", "api_key", "credential",
    "access_token", "private_key", "clipboard", "clipboard_data",
}


def sanitize_event_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Redact secrets, credentials, raw screenshots, and unbounded payloads from event metadata.
    """
    if not metadata:
        return {}
    sanitized: Dict[str, Any] = {}
    for k, v in metadata.items():
        k_lower = k.lower()
        if any(s in k_lower for s in _SENSITIVE_KEYS):
            sanitized[k] = "[REDACTED]"
        elif k_lower in ("screenshot", "image", "image_data", "raw_image") and isinstance(v, (str, bytes)):
            sanitized[k] = f"[IMAGE DATA: length={len(v)}]"
        elif k_lower in ("text", "typed_text", "input_text") and isinstance(v, str) and len(v) > 200:
            sanitized[k] = f"{v[:100]}... [truncated {len(v)} chars]"
        elif isinstance(v, dict):
            sanitized[k] = sanitize_event_metadata(v)
        elif isinstance(v, (list, tuple)):
            sanitized[k] = [
                sanitize_event_metadata(item) if isinstance(item, dict) else item
                for item in v[:20]
            ]
        else:
            sanitized[k] = v
    return sanitized


@dataclass(frozen=True)
class CognitiveEvent:
    """
    Immutable structured domain event capturing an observable step in the cognitive lifecycle.
    Never stores raw passwords, credentials, full hidden prompts, or unrestricted screenshots.
    """
    event_id: str
    turn_id: str
    session_id: str
    stage: CognitiveStage
    event_type: CognitiveEventType
    timestamp: float
    duration: float = 0.0
    status: str = "OK"
    component: str = "runtime"
    summary: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_event_id: Optional[str] = None

    def __post_init__(self) -> None:
        # Enforce metadata immutability and sanitation
        if self.metadata:
            sanitized = sanitize_event_metadata(self.metadata)
            object.__setattr__(self, "metadata", dict(sanitized))


@dataclass(frozen=True)
class CognitiveTrace:
    """
    Bounded, immutable trace representation for a single cognitive turn.
    Guarantees event history never grows unbounded in memory.
    """
    turn_id: str
    session_id: str
    events: Tuple[CognitiveEvent, ...] = ()
    start_time: float = 0.0
    end_time: Optional[float] = None
    final_status: TurnStatus = TurnStatus.RUNNING
    max_events: int = 100

    @property
    def event_count(self) -> int:
        return len(self.events)

    @property
    def elapsed_time(self) -> float:
        if self.end_time is not None:
            return max(0.0, self.end_time - self.start_time)
        return max(0.0, time.time() - self.start_time)


@dataclass(frozen=True)
class TurnLimits:
    """
    Safety limits and timeouts for a cognitive turn.
    """
    max_duration_seconds: float = 60.0
    max_events: int = 100
    max_recovery_attempts: int = 3
    max_execution_steps: int = 20


@dataclass
class CognitiveTurn:
    """
    State tracking entity representing one ATLAS cognitive turn.
    Stores bounded snapshots/references without unbounded historical accumulation.
    """
    turn_id: str
    session_id: str
    original_request: Optional[Request] = None
    original_goal: str = ""
    current_stage: CognitiveStage = CognitiveStage.RECEIVED
    status: TurnStatus = TurnStatus.RUNNING
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    current_task: Optional[Task] = None
    decision: Optional[Decision] = None
    current_plan: Optional[Plan] = None
    current_context: Optional[ContextSelection] = None
    routing_result: Optional[RoutingResult] = None
    reasoning_result: Optional[ReasoningResponse] = None
    proposal_result: Optional[ProposalValidationResult] = None
    policy_result: Optional[PolicyResult] = None
    execution_results: List[Result] = field(default_factory=list)
    verification_result: Optional[VerificationResult] = None
    recovery_result: Optional[RecoveryContext] = None
    final_response: Optional[str] = None
    waiting_reason: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        return self.status in (TurnStatus.RUNNING, TurnStatus.WAITING_FOR_USER)

    @property
    def elapsed_seconds(self) -> float:
        if self.end_time is not None:
            return max(0.0, self.end_time - self.start_time)
        return max(0.0, time.time() - self.start_time)


@dataclass(frozen=True)
class CognitiveTurnResult:
    """
    Final structured, immutable result of a completed or stopped cognitive turn.
    Provides backward-compatible conversion to PipelineResult.
    """
    turn_id: str
    session_id: str
    status: TurnStatus
    stage: CognitiveStage
    response: str
    trace: CognitiveTrace
    request: Optional[Request] = None
    decision: Optional[Decision] = None
    plan: Optional[Plan] = None
    results: List[Result] = field(default_factory=list)
    verification: Optional[VerificationResult] = None
    recovery: Optional[RecoveryContext] = None
    waiting_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_pipeline_result(self) -> PipelineResult:
        """
        Convert to standard PipelineResult for backward compatibility with existing callers.
        """
        req = self.request or Request(
            id=self.turn_id,
            session_id=self.session_id,
            original_text=self.response,
        )
        dec = self.decision or Decision(
            request_id=req.id,
            primary_goal=req.original_text,
            required_capabilities=[],
            execution_mode=ExecutionMode.DIRECT,
            confidence=1.0,
            reasoning="Constructed fallback decision",
        )
        pln = self.plan or Plan(goal=req.original_text, steps=[])
        ver = self.verification or VerificationResult(
            verified=(self.status == TurnStatus.SUCCEEDED),
            status="verified" if self.status == TurnStatus.SUCCEEDED else "failed",
            confidence=1.0,
            reason="Synthesized verification for turn",
        )
        return PipelineResult(
            response=self.response,
            request=req,
            decision=dec,
            plan=pln,
            results=self.results,
            verification=ver,
            recovery=self.recovery,
        )
