import logging
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.models.runtime import (
    CognitiveEvent,
    CognitiveEventType,
    CognitiveStage,
    CognitiveTrace,
    CognitiveTurnResult,
    TurnLimits,
    TurnStatus,
)
from core.models.replay import (
    ComparisonResult,
    ReplayDivergence,
    ReplayLimits,
    ReplayMode,
    ReplayRequest,
    ReplayResult,
)
from core.models.request import Request
from core.models.decision import Decision, CapabilityType, ExecutionMode
from core.models.plan import Plan
from core.models.task import Task
from core.models.result import Result
from core.models.policy import PolicyDecision, PolicyResult, PolicyContext
from core.models.tool_call import ToolCall
from core.models.verification import VerificationResult
from core.models.reasoning import ReasoningResponse, ReasoningOutcome, ActionProposal, ProposalValidationResult
from core.models.memory import MessageRole, ChatMessage, MemoryEntry

from core.interfaces.execution_engine_interface import ExecutionEngineInterface
from core.interfaces.policy_interface import PolicyEngineInterface
from core.interfaces.memory_interface import MemoryServiceInterface
from core.interfaces.verification_interface import VerificationInterface
from core.interfaces.response_composer_interface import ResponseComposerInterface

from runtime.cognitive_runtime import CognitiveRuntime
from runtime.event_sink import InMemoryEventSink
from runtime.recorded_store import RecordedResultStore
from runtime.comparator import TraceComparator

logger = logging.getLogger("atlas.replay_engine")


class ReplaySafetyViolation(RuntimeError):
    """Raised when a replay run attempts an unauthorized live operation."""
    pass


class ReplayExecutionEngine(ExecutionEngineInterface):
    """
    Replay-safe ExecutionEngine wrapper.
    Resolves tasks strictly via:
    1. Recorded results from RecordedResultStore
    2. Explicit safe simulation provider
    NEVER invokes real system, native computer, shell, or production tool orchestrator.
    """

    def __init__(
        self,
        source_turn_id: str,
        recorded_store: RecordedResultStore,
        simulation_provider: Optional[Callable[[Task], Result]] = None,
        replay_mode: ReplayMode = ReplayMode.OFFLINE_REPLAY,
    ):
        self.source_turn_id = source_turn_id
        self.recorded_store = recorded_store
        self.simulation_provider = simulation_provider
        self.replay_mode = replay_mode
        self.executed_tasks: List[Task] = []

    def execute(self, plan: Plan) -> List[Result]:
        results: List[Result] = []
        for idx, step in enumerate(plan.steps, 1):
            self.executed_tasks.append(step)
            call_id = f"call_{self.source_turn_id}_{getattr(step, 'id', idx)}"
            step_key = f"step_{idx}"

            # 1. In SIMULATION_REPLAY mode, prefer simulation provider if present
            if self.replay_mode == ReplayMode.SIMULATION_REPLAY and self.simulation_provider is not None:
                try:
                    sim_res = self.simulation_provider(step)
                    results.append(sim_res)
                    continue
                except Exception as ex:
                    results.append(
                        Result(
                            success=False,
                            message=f"Simulation provider error: {ex}",
                            call_id=call_id,
                        )
                    )
                    continue

            # 2. Check recorded result store
            rec_res = (
                self.recorded_store.get_tool_result(self.source_turn_id, call_id)
                or self.recorded_store.get_tool_result(self.source_turn_id, step_key)
                or self.recorded_store.get_tool_result(self.source_turn_id, str(getattr(step, "id", idx)))
                or self.recorded_store.get_tool_result(self.source_turn_id, f"call_{getattr(step, 'id', idx)}")
            )

            if rec_res is not None:
                results.append(rec_res)
                continue

            # 3. Fallback to simulation provider
            if self.simulation_provider is not None:
                try:
                    sim_res = self.simulation_provider(step)
                    results.append(sim_res)
                    continue
                except Exception as ex:
                    results.append(
                        Result(
                            success=False,
                            message=f"Simulation provider error: {ex}",
                            call_id=call_id,
                        )
                    )
                    continue

            # 3. Neither exists: Fail safe. Never call real production tools.
            results.append(
                Result(
                    success=False,
                    message=f"Replay missing recorded result for action '{getattr(step, 'action', 'unknown')}'; live execution prevented.",
                    output=None,
                    call_id=call_id,
                )
            )

        return results


class ReplayMemoryService(MemoryServiceInterface):
    """
    Isolated in-memory MemoryService fake for replay turns.
    Guarantees replay turns NEVER mutate or overwrite production SQLite memory.
    """

    def __init__(self):
        self.messages: List[ChatMessage] = []
        self.preferences: Dict[Tuple[str, str], MemoryEntry] = {}

    def add_message(
        self,
        session_id: str,
        role: MessageRole,
        content: str,
        metadata: Optional[dict] = None,
        message_id: Optional[str] = None,
    ) -> ChatMessage:
        msg = ChatMessage(
            id=message_id or f"msg_replay_{uuid.uuid4().hex[:8]}",
            session_id=session_id,
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {},
        )
        self.messages.append(msg)
        return msg

    def get_history(self, session_id: str, limit: Optional[int] = None) -> List[ChatMessage]:
        matching = [m for m in self.messages if m.session_id == session_id]
        if limit is not None:
            return matching[-limit:]
        return matching

    def clear_session(self, session_id: str) -> None:
        self.messages = [m for m in self.messages if m.session_id != session_id]

    def save_preference(self, user_id: str, key: str, value: Any, category: str = "general") -> MemoryEntry:
        entry = MemoryEntry(user_id=user_id, key=key, value=value, category=category)
        self.preferences[(user_id, key)] = entry
        return entry

    def get_preference(self, user_id: str, key: str) -> Optional[MemoryEntry]:
        return self.preferences.get((user_id, key))

    def delete_preference(self, user_id: str, key: str) -> bool:
        if (user_id, key) in self.preferences:
            del self.preferences[(user_id, key)]
            return True
        return False

    def list_preferences(self, user_id: str) -> List[MemoryEntry]:
        return [e for (uid, _), e in self.preferences.items() if uid == user_id]


class ReplayPolicyEngine(PolicyEngineInterface):
    """
    Policy engine adapter for replay.
    Supports either:
    1. Using recorded policy decision (when policy_reevaluation is False)
    2. Re-evaluating current live policy (when policy_reevaluation is True)
    """

    def __init__(
        self,
        source_turn_id: str,
        recorded_store: RecordedResultStore,
        live_policy_engine: Optional[PolicyEngineInterface] = None,
        reevaluate: bool = False,
    ):
        self.source_turn_id = source_turn_id
        self.recorded_store = recorded_store
        self.live_policy_engine = live_policy_engine
        self.reevaluate = reevaluate
        self.evaluation_count = 0

    def evaluate(self, tool_call: ToolCall, context: PolicyContext) -> PolicyResult:
        self.evaluation_count += 1
        step_key = f"step_{self.evaluation_count}"

        if self.reevaluate and self.live_policy_engine is not None:
            # Re-evaluate live policy
            return self.live_policy_engine.evaluate(tool_call, context)

        # Use recorded policy decision if available
        rec_pol = (
            self.recorded_store.get_policy_result(self.source_turn_id, step_key)
            or self.recorded_store.get(f"policy:{self.source_turn_id}:latest")
        )

        if rec_pol is not None:
            return rec_pol

        # Default fallback if recorded policy not found
        if self.live_policy_engine is not None:
            return self.live_policy_engine.evaluate(tool_call, context)

        return PolicyResult(
            decision=PolicyDecision.ALLOW,
            rule_id="replay_default",
            reason="Replay allowed by default",
        )


class ReplayReasoningEngine:
    """
    Reasoning engine adapter for replay.
    Returns recorded reasoning response or simulation provider without calling live models.
    """

    def __init__(
        self,
        source_turn_id: str,
        recorded_store: RecordedResultStore,
        mock_provider: Optional[Any] = None,
    ):
        self.source_turn_id = source_turn_id
        self.recorded_store = recorded_store
        self.mock_provider = mock_provider

    def run_turn(
        self,
        goal: str,
        task_description: Optional[str] = None,
        context_selection: Optional[Any] = None,
    ) -> Tuple[ReasoningResponse, Optional[ProposalValidationResult]]:
        # 1. Check recorded store
        rec_reasoning = self.recorded_store.get_reasoning_result(self.source_turn_id)
        if rec_reasoning is not None:
            val = ProposalValidationResult(is_valid=True) if rec_reasoning.proposal else None
            return rec_reasoning, val

        # 2. Check mock provider
        if self.mock_provider is not None and hasattr(self.mock_provider, "run_turn"):
            return self.mock_provider.run_turn(goal, task_description, context_selection)

        # 3. Default fallback: synthesize empty/clarification response
        resp = ReasoningResponse(
            turn_id=f"reason_{self.source_turn_id}",
            outcome=ReasoningOutcome.ABORT,
            abort_reason="No recorded reasoning response found in replay store",
            confidence=1.0,
        )
        return resp, None


class ReplayVerifier(VerificationInterface):
    """
    Replay-safe verifier adapter.
    Uses recorded verification result if available, or delegates to provided verifier.
    """

    def __init__(
        self,
        source_turn_id: str,
        recorded_store: RecordedResultStore,
        fallback_verifier: Optional[VerificationInterface] = None,
    ):
        self.source_turn_id = source_turn_id
        self.recorded_store = recorded_store
        self.fallback_verifier = fallback_verifier

    def verify(self, plan: Plan, results: List[Result]) -> VerificationResult:
        if self.fallback_verifier is not None:
            return self.fallback_verifier.verify(plan, results)
        rec_ver = self.recorded_store.get(f"verification:{self.source_turn_id}")
        if rec_ver is not None and isinstance(rec_ver, VerificationResult):
            return rec_ver
        from brain.verification import StandardVerifier
        return StandardVerifier().verify(plan, results)


class ReplayComposer(ResponseComposerInterface):
    """
    Replay-safe response composer adapter.
    Uses recorded final response if available or delegates to fallback composer.
    """

    def __init__(
        self,
        source_turn_id: str,
        recorded_store: RecordedResultStore,
        fallback_composer: Optional[ResponseComposerInterface] = None,
    ):
        self.source_turn_id = source_turn_id
        self.recorded_store = recorded_store
        self.fallback_composer = fallback_composer

    def compose(
        self,
        request: Request,
        decision: Decision,
        plan: Plan,
        results: List[Result],
        verification: VerificationResult,
    ) -> str:
        if self.fallback_composer is not None:
            return self.fallback_composer.compose(request, decision, plan, results, verification)
        rec_resp = self.recorded_store.get(f"response:{self.source_turn_id}")
        if rec_resp is not None and isinstance(rec_resp, str):
            return rec_resp
        from brain.response import StandardResponseComposer
        return StandardResponseComposer().compose(request, decision, plan, results, verification)


class ReplayEngine:
    """
    Engine for executing deterministic, bounded, and side-effect-free replay
    of historical ATLAS cognitive turns.
    Coordinates standard CognitiveRuntime with sandboxed execution, reasoning,
    memory, and policy boundaries.
    """

    def __init__(
        self,
        recorded_store: Optional[RecordedResultStore] = None,
    ):
        self.recorded_store = recorded_store or RecordedResultStore()

    def replay(self, request: ReplayRequest) -> ReplayResult:
        """
        Execute deterministic replay for a given ReplayRequest.
        """
        source_trace = request.source_trace
        turn_id = source_trace.turn_id
        session_id = source_trace.session_id
        replay_run_id = request.replay_run_id

        # 1. Populate recorded store from source trace
        self.recorded_store.extract_from_trace(source_trace)

        # 2. Extract input text from source trace
        input_text = self._extract_request_text(source_trace)

        # 3. Configure sandboxed isolated components
        sink = InMemoryEventSink()
        limits = TurnLimits(
            max_duration_seconds=30.0,
            max_events=request.limits.max_events,
            max_execution_steps=request.limits.max_replay_steps,
        )

        overrides = request.component_overrides
        sim_exec = overrides.get("simulation_provider")
        live_policy = overrides.get("policy_engine")
        mock_reasoning = overrides.get("reasoning_provider")

        exec_engine = ReplayExecutionEngine(
            source_turn_id=turn_id,
            recorded_store=self.recorded_store,
            simulation_provider=sim_exec,
            replay_mode=request.replay_mode,
        )

        mem_service = overrides.get("memory_service") or ReplayMemoryService()

        # Check if source trace used policy engine
        has_policy_event = any(ev.stage == CognitiveStage.POLICY for ev in source_trace.events)
        policy_engine = None
        if has_policy_event or live_policy is not None or request.policy_reevaluation:
            policy_engine = ReplayPolicyEngine(
                source_turn_id=turn_id,
                recorded_store=self.recorded_store,
                live_policy_engine=live_policy,
                reevaluate=request.policy_reevaluation,
            )

        # Optional reasoning engine
        reasoning_engine = None
        has_reasoning_event = any(
            ev.event_type == CognitiveEventType.REASONING_COMPLETED
            for ev in source_trace.events
        )
        if has_reasoning_event or mock_reasoning:
            reasoning_engine = ReplayReasoningEngine(
                source_turn_id=turn_id,
                recorded_store=self.recorded_store,
                mock_provider=mock_reasoning,
            )

        # Build sandboxed CognitiveRuntime
        runtime = CognitiveRuntime(
            understanding=overrides.get("understanding"),
            decision_engine=overrides.get("decision_engine"),
            planner=overrides.get("planner"),
            context_manager=overrides.get("context_manager"),
            model_router=overrides.get("model_router"),
            reasoning_engine=reasoning_engine,
            policy_engine=policy_engine,
            execution_engine=exec_engine,
            verifier=overrides.get("verifier") or ReplayVerifier(turn_id, self.recorded_store),
            recovery_engine=overrides.get("recovery_engine"),
            memory_service=mem_service,
            composer=overrides.get("composer") or ReplayComposer(turn_id, self.recorded_store),
            event_sink=sink,
            limits=limits,
        )

        # 4. Execute sandboxed turn
        completed_stages: List[CognitiveStage] = []
        turn_result: Optional[CognitiveTurnResult] = None
        try:
            turn_result = runtime.execute_turn(
                input_data=input_text,
                session_id=f"replay_{session_id}",
            )
            replay_trace = turn_result.trace
            completed_stages = [ev.stage for ev in replay_trace.events if ev.stage not in completed_stages]
        except Exception as ex:
            logger.error("Error during replay execution: %s", ex, exc_info=True)
            # Build minimal failed trace from captured events
            captured_events = sink.get_events()
            replay_trace = CognitiveTrace(
                turn_id=replay_run_id,
                session_id=f"replay_{session_id}",
                events=tuple(captured_events),
                final_status=TurnStatus.FAILED,
            )

        # 5. Compare source trace vs replay trace
        comparison = TraceComparator.compare(
            original_trace=source_trace,
            replay_trace=replay_trace,
            max_divergences=request.limits.max_divergences,
        )

        success = comparison.is_equivalent and (
            turn_result is None or turn_result.status != TurnStatus.FAILED
        )

        return ReplayResult(
            success=success,
            replay_run_id=replay_run_id,
            source_turn_id=turn_id,
            completed_stages=tuple(completed_stages),
            replay_trace=replay_trace,
            comparison=comparison,
            final_status=replay_trace.final_status,
            diagnostics={
                "matched_events": comparison.matched_events_count,
                "divergence_count": len(comparison.divergences),
                "summary": comparison.summary,
            },
        )

    def _extract_request_text(self, trace: CognitiveTrace) -> str:
        """Infer or extract original user prompt text from trace events."""
        for ev in trace.events:
            if ev.event_type == CognitiveEventType.STAGE_STARTED and ev.stage == CognitiveStage.UNDERSTANDING:
                if ev.summary and "for request: " in ev.summary:
                    return ev.summary.split("for request: ", 1)[1]
            if ev.event_type == CognitiveEventType.TURN_STARTED and ev.metadata:
                if "goal" in ev.metadata:
                    return str(ev.metadata["goal"])
                if "input" in ev.metadata:
                    return str(ev.metadata["input"])
        # Fallback to goal reference in metadata or first event summary
        if trace.events:
            return trace.events[0].summary or "Replayed query"
        return "Replayed query"
