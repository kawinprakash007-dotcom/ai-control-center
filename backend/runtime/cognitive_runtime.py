import logging
import time
from typing import Any, Dict, List, Optional
import uuid

from core.interfaces.runtime_interface import (
    CognitiveRuntimeInterface,
    CognitiveEventSinkInterface,
)
from core.interfaces.request_understanding_interface import RequestUnderstandingInterface
from core.interfaces.decision_engine_interface import DecisionEngineInterface
from core.interfaces.decision_planner_interface import DecisionPlannerInterface
from core.interfaces.execution_engine_interface import ExecutionEngineInterface
from core.interfaces.verification_interface import VerificationInterface
from core.interfaces.response_composer_interface import ResponseComposerInterface
from core.interfaces.memory_interface import MemoryServiceInterface
from core.interfaces.web_interface import WebProviderInterface
from core.interfaces.recovery_interface import RecoveryEngineInterface
from core.interfaces.policy_interface import PolicyEngineInterface
from core.interfaces.context_interface import ContextManagerInterface
from core.interfaces.model_router_interface import ModelRouterInterface

from core.models.runtime import (
    CognitiveStage,
    TurnStatus,
    CognitiveEventType,
    CognitiveEvent,
    CognitiveTrace,
    CognitiveTurn,
    CognitiveTurnResult,
    TurnLimits,
)
from core.models.request import Request
from core.models.decision import Decision
from core.models.plan import Plan
from core.models.result import Result
from core.models.verification import VerificationResult
from core.models.memory import MessageRole
from core.models.tool_call import ToolCall
from core.models.policy import PolicyContext, PolicyDecision
from core.models.recovery import RecoveryAction
from core.models.context import CognitiveState
from core.models.reasoning import ReasoningOutcome

from runtime.transitions import validate_stage_transition, InvalidTransitionError
from runtime.event_sink import InMemoryEventSink

logger = logging.getLogger("atlas.cognitive_runtime")


class CognitiveRuntime(CognitiveRuntimeInterface):
    """
    ATLAS Unified Cognitive Runtime.
    Orchestrates the Phase 2 and Phase 3 cognitive lifecycle into one deterministic,
    observable, and bounded control pipeline:
    RECEIVED -> UNDERSTANDING -> DECISION -> PLANNING -> CONTEXT -> ROUTING ->
    REASONING -> PROPOSAL -> VALIDATION -> POLICY -> EXECUTION -> OBSERVATION ->
    VERIFICATION -> (RECOVERY) -> MEMORY -> RESPONSE -> COMPLETED.

    Architectural Safeguards:
    1. Thin coordination boundary: does NOT implement tool execution, model reasoning,
       policy rules, or memory persistence.
    2. Explicit state transitions: guarded by validate_stage_transition.
    3. Structured, bounded, privacy-preserving event sink.
    4. First-class WAITING_FOR_USER handling for permissions, confirmations, and clarifications.
    5. Clean error boundaries: unexpected errors mark turn FAILED without exception hiding.
    """

    def __init__(
        self,
        understanding: Optional[RequestUnderstandingInterface] = None,
        decision_engine: Optional[DecisionEngineInterface] = None,
        planner: Optional[DecisionPlannerInterface] = None,
        context_manager: Optional[ContextManagerInterface] = None,
        model_router: Optional[ModelRouterInterface] = None,
        reasoning_engine: Optional[Any] = None,
        policy_engine: Optional[PolicyEngineInterface] = None,
        execution_engine: Optional[ExecutionEngineInterface] = None,
        verifier: Optional[VerificationInterface] = None,
        recovery_engine: Optional[RecoveryEngineInterface] = None,
        memory_service: Optional[MemoryServiceInterface] = None,
        composer: Optional[ResponseComposerInterface] = None,
        web_provider: Optional[WebProviderInterface] = None,
        event_sink: Optional[CognitiveEventSinkInterface] = None,
        limits: Optional[TurnLimits] = None,
    ):
        # Default component wiring matching standard architecture
        if understanding is None:
            from brain.request_understanding import StandardRequestUnderstanding
            self.understanding = StandardRequestUnderstanding()
        else:
            self.understanding = understanding

        if decision_engine is None:
            from brain.decision_engine import StandardDecisionEngine
            self.decision_engine = StandardDecisionEngine()
        else:
            self.decision_engine = decision_engine

        if planner is None:
            from brain.planning import StandardPlanner
            self.planner = StandardPlanner()
        else:
            self.planner = planner

        if execution_engine is None:
            from brain.execution import StandardExecutionEngine
            self.execution_engine = StandardExecutionEngine()
        else:
            self.execution_engine = execution_engine

        if verifier is None:
            from brain.verification import StandardVerifier
            self.verifier = StandardVerifier()
        else:
            self.verifier = verifier

        if composer is None:
            from brain.response import StandardResponseComposer
            self.composer = StandardResponseComposer()
        else:
            self.composer = composer

        if memory_service is None:
            from memory.sqlite_store import SQLiteMemoryStore
            self.memory_service = SQLiteMemoryStore()
        else:
            self.memory_service = memory_service

        self.web_provider = web_provider
        self.context_manager = context_manager
        self.model_router = model_router
        self.reasoning_engine = reasoning_engine
        self.policy_engine = policy_engine
        self.recovery_engine = recovery_engine
        self.event_sink = event_sink or InMemoryEventSink()
        self.limits = limits or TurnLimits()

    def _publish_event(
        self,
        turn: CognitiveTurn,
        event_type: CognitiveEventType,
        stage: Optional[CognitiveStage] = None,
        status: str = "OK",
        component: str = "runtime",
        summary: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CognitiveEvent:
        """Publish an immutable structured cognitive event."""
        event = CognitiveEvent(
            event_id=f"evt_{uuid.uuid4().hex[:10]}",
            turn_id=turn.turn_id,
            session_id=turn.session_id,
            stage=stage or turn.current_stage,
            event_type=event_type,
            timestamp=time.time(),
            duration=turn.elapsed_seconds,
            status=status,
            component=component,
            summary=summary,
            metadata=metadata or {},
        )
        self.event_sink.publish(event)
        return event

    def _advance_stage(
        self,
        turn: CognitiveTurn,
        target_stage: CognitiveStage,
        summary: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Deterministically advance lifecycle stage with graph validation."""
        validate_stage_transition(turn.current_stage, target_stage)
        turn.current_stage = target_stage
        self._publish_event(
            turn=turn,
            event_type=CognitiveEventType.STAGE_STARTED,
            stage=target_stage,
            summary=summary or f"Stage advanced to {target_stage.value}",
            metadata=metadata or {},
        )

    def _check_deadline(self, turn: CognitiveTurn, max_duration: float) -> bool:
        """Check if turn exceeded deadline. Returns True if exceeded."""
        if turn.elapsed_seconds > max_duration:
            turn.status = TurnStatus.ABORTED
            turn.error = f"Turn exceeded maximum duration of {max_duration:.1f}s."
            return True
        return False

    def _inject_context(
        self,
        plan: Plan,
        request: Request,
        history_entries: List[Any],
        is_empty: bool,
    ) -> None:
        """Inject contextual dependencies into plan steps without mutating Request."""
        if not plan.steps or is_empty:
            return

        formatted_history: List[Dict[str, str]] = [
            {"role": msg.role.value if hasattr(msg.role, "value") else str(msg.role), "content": msg.content}
            for msg in history_entries
        ]
        for step in plan.steps:
            step_type = getattr(step, "type", None)
            step_tool = getattr(step, "tool", None)
            if step_type == "chat" or step_tool == "chat":
                params = dict(step.parameters) if step.parameters else {}
                if "query" not in params:
                    params["query"] = request.original_text
                if "history" not in params:
                    params["history"] = formatted_history
                step.parameters = params
            elif step_type == "memory" or step_tool == "memory":
                params = dict(step.parameters) if step.parameters else {}
                if "memory_service" not in params and self.memory_service is not None:
                    params["memory_service"] = self.memory_service
                if "user_id" not in params:
                    params["user_id"] = "default_user"
                step.parameters = params
            elif step_type == "web" or step_tool == "web":
                params = dict(step.parameters) if step.parameters else {}
                if "web_provider" not in params and self.web_provider is not None:
                    params["web_provider"] = self.web_provider
                step.parameters = params

    def execute_turn(
        self,
        input_data: Any,
        session_id: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ) -> CognitiveTurnResult:
        """
        Execute one complete ATLAS cognitive turn through the unified lifecycle.
        """
        turn_id = f"turn_{uuid.uuid4().hex[:12]}"
        effective_session_id = session_id or "default_session"
        max_duration = timeout_seconds if timeout_seconds is not None else self.limits.max_duration_seconds

        turn = CognitiveTurn(
            turn_id=turn_id,
            session_id=effective_session_id,
            current_stage=CognitiveStage.RECEIVED,
            status=TurnStatus.RUNNING,
            start_time=time.time(),
        )

        self._publish_event(
            turn=turn,
            event_type=CognitiveEventType.TURN_STARTED,
            stage=CognitiveStage.RECEIVED,
            summary="Cognitive turn started",
            metadata={"session_id": effective_session_id, "input": str(input_data)},
        )

        is_empty = False
        user_message_persisted = False
        assistant_message_persisted = False

        try:
            # -------------------------------------------------------------
            # STAGE 1: UNDERSTANDING
            # -------------------------------------------------------------
            self._advance_stage(turn, CognitiveStage.UNDERSTANDING)
            if self._check_deadline(turn, max_duration):
                return self._finalize_aborted(turn)

            request = self.understanding.understand(input_data)
            turn.original_request = request
            turn.original_goal = request.original_text
            if not session_id and request.session_id:
                turn.session_id = request.session_id

            self._publish_event(
                turn=turn,
                event_type=CognitiveEventType.STAGE_COMPLETED,
                stage=CognitiveStage.UNDERSTANDING,
                summary="Request understood successfully",
                metadata={"request_id": request.id},
            )

            is_empty = bool(request.parameters.get("is_empty", False))

            # -------------------------------------------------------------
            # STAGE 2: DECISION
            # -------------------------------------------------------------
            self._advance_stage(turn, CognitiveStage.DECISION)
            if self._check_deadline(turn, max_duration):
                return self._finalize_aborted(turn)

            # Read conversational memory
            history_entries: List[Any] = []
            if not is_empty and self.memory_service is not None:
                history_entries = self.memory_service.get_history(turn.session_id, limit=20)

            decision = self.decision_engine.decide(request)
            turn.decision = decision
            turn.metadata["decision"] = decision

            self._publish_event(
                turn=turn,
                event_type=CognitiveEventType.DECISION_MADE,
                stage=CognitiveStage.DECISION,
                summary=f"Decision made: intent={getattr(decision, 'intent', None)}",
                metadata={
                    "intent": str(getattr(decision, "intent", "")),
                    "capability": str(getattr(decision, "capability", "")),
                },
            )

            # -------------------------------------------------------------
            # STAGE 3: PLANNING
            # -------------------------------------------------------------
            self._advance_stage(turn, CognitiveStage.PLANNING)
            if self._check_deadline(turn, max_duration):
                return self._finalize_aborted(turn)

            # Pre-execution planning: do not catch ValueError as caller may rely on pre-execution fallback
            plan = self.planner.plan(decision)
            turn.current_plan = plan

            self._publish_event(
                turn=turn,
                event_type=CognitiveEventType.PLAN_CREATED,
                stage=CognitiveStage.PLANNING,
                summary=f"Plan created with {len(plan.steps)} steps",
                metadata={"plan_id": getattr(plan, "id", f"plan_{turn.turn_id}"), "step_count": len(plan.steps)},
            )

            # -------------------------------------------------------------
            # STAGE 4: CONTEXT (Phase 3.9)
            # -------------------------------------------------------------
            if self.context_manager is not None:
                self._advance_stage(turn, CognitiveStage.CONTEXT)
                if self._check_deadline(turn, max_duration):
                    return self._finalize_aborted(turn)

                step_task_desc = getattr(plan.steps[0], "action", None) or getattr(plan.steps[0], "description", None) if plan.steps else None
                cs = CognitiveState(
                    original_goal=request.original_text,
                    current_task=step_task_desc,
                    memory_items=tuple(str(getattr(m, "content", m)) for m in history_entries),
                )
                context_selection = self.context_manager.build_context(cs)
                turn.current_context = context_selection
                self._publish_event(
                    turn=turn,
                    event_type=CognitiveEventType.CONTEXT_SELECTED,
                    stage=CognitiveStage.CONTEXT,
                    summary=f"Context selected ({len(context_selection.selected_items)} items)",
                    metadata={
                        "item_count": len(context_selection.selected_items),
                        "attention": context_selection.attention_focus.value,
                    },
                )

            # -------------------------------------------------------------
            # STAGE 5: MODEL ROUTING (Phase 3.7)
            # -------------------------------------------------------------
            if self.model_router is not None:
                self._advance_stage(turn, CognitiveStage.ROUTING)
                if self._check_deadline(turn, max_duration):
                    return self._finalize_aborted(turn)

                try:
                    from routing.requirement_mapper import derive_requirements_from_request
                    from core.models.reasoning import ReasoningRequest
                    step_task_desc = getattr(plan.steps[0], "action", None) or getattr(plan.steps[0], "description", None) if plan.steps else None
                    r_req = ReasoningRequest(
                        goal=request.original_text,
                        task_description=step_task_desc,
                    )
                    requirements = derive_requirements_from_request(r_req)
                    routing_res = self.model_router.route(requirements)
                    turn.routing_result = routing_res
                    self._publish_event(
                        turn=turn,
                        event_type=CognitiveEventType.MODEL_ROUTED,
                        stage=CognitiveStage.ROUTING,
                        summary=f"Model routed: provider={routing_res.provider_id if routing_res.success else 'failed'}",
                        metadata={"provider_id": routing_res.provider_id if routing_res.success else None},
                    )
                except Exception as e:
                    logger.debug("Optional model routing skipped or failed: %s", e)

            # -------------------------------------------------------------
            # STAGE 6: REASONING & PROPOSAL (Phase 3.6)
            # -------------------------------------------------------------
            if self.reasoning_engine is not None:
                self._advance_stage(turn, CognitiveStage.REASONING)
                if self._check_deadline(turn, max_duration):
                    return self._finalize_aborted(turn)

                step_task_desc = getattr(plan.steps[0], "action", None) or getattr(plan.steps[0], "description", None) if plan.steps else None
                reasoning_resp, validation_res = self.reasoning_engine.run_turn(
                    goal=request.original_text,
                    task_description=step_task_desc,
                    context_selection=turn.current_context,
                )
                turn.reasoning_result = reasoning_resp
                self._publish_event(
                    turn=turn,
                    event_type=CognitiveEventType.REASONING_COMPLETED,
                    stage=CognitiveStage.REASONING,
                    summary=f"Reasoning completed: outcome={reasoning_resp.outcome.value}",
                    metadata={"outcome": reasoning_resp.outcome.value},
                )

                if reasoning_resp.outcome == ReasoningOutcome.PROPOSE_ACTION:
                    self._advance_stage(turn, CognitiveStage.PROPOSAL)
                    prop = getattr(reasoning_resp, "proposal", None) or getattr(reasoning_resp, "action_proposal", None)
                    self._publish_event(
                        turn=turn,
                        event_type=CognitiveEventType.PROPOSAL_CREATED,
                        stage=CognitiveStage.PROPOSAL,
                        summary="Action proposed by reasoning layer",
                        metadata={
                            "proposal_id": prop.proposal_id if prop else None
                        },
                    )
                    self._advance_stage(turn, CognitiveStage.VALIDATION)
                    turn.proposal_result = validation_res
                    is_valid = bool(validation_res and (getattr(validation_res, "is_valid", False) or getattr(validation_res, "valid", False)))
                    if is_valid:
                        self._publish_event(
                            turn=turn,
                            event_type=CognitiveEventType.PROPOSAL_VALIDATED,
                            stage=CognitiveStage.VALIDATION,
                            summary="Action proposal validated successfully",
                        )
                    else:
                        self._publish_event(
                            turn=turn,
                            event_type=CognitiveEventType.PROPOSAL_REJECTED,
                            stage=CognitiveStage.VALIDATION,
                            status="REJECTED",
                            summary="Action proposal validation failed",
                        )

                elif reasoning_resp.outcome == ReasoningOutcome.ASK_CLARIFICATION:
                    turn.status = TurnStatus.WAITING_FOR_USER
                    turn.waiting_reason = (
                        getattr(reasoning_resp, "clarification_prompt", None)
                        or getattr(reasoning_resp, "clarification_question", None)
                        or "Clarification requested"
                    )
                    self._advance_stage(turn, CognitiveStage.RESPONSE)
                    turn.final_response = turn.waiting_reason
                    self._publish_event(
                        turn=turn,
                        event_type=CognitiveEventType.TURN_WAITING,
                        stage=CognitiveStage.RESPONSE,
                        summary=f"Turn waiting for clarification: {turn.waiting_reason}",
                    )
                    return self._build_result(turn, decision=decision, plan=plan)

                elif reasoning_resp.outcome == ReasoningOutcome.ABORT:
                    turn.status = TurnStatus.ABORTED
                    turn.error = getattr(reasoning_resp, "abort_reason", "Reasoning layer requested abort")
                    self._advance_stage(turn, CognitiveStage.ABORTED)
                    self._publish_event(
                        turn=turn,
                        event_type=CognitiveEventType.TURN_ABORTED,
                        stage=CognitiveStage.ABORTED,
                        summary=f"Reasoning layer aborted turn: {turn.error}",
                    )
                    return self._build_result(turn, decision=decision, plan=plan)

            # -------------------------------------------------------------
            # STAGE 7: POLICY (Phase 3.1)
            # -------------------------------------------------------------
            if self.policy_engine is not None:
                self._advance_stage(turn, CognitiveStage.POLICY)
                if self._check_deadline(turn, max_duration):
                    return self._finalize_aborted(turn)

                policy_blocked = False
                for step in plan.steps:
                    tc = ToolCall(
                        capability=getattr(step, "tool", None) or getattr(step, "type", "tool"),
                        action=getattr(step, "type", "execute"),
                        parameters=getattr(step, "parameters", {}) or {},
                        call_id=f"call_{turn.turn_id}_{getattr(step, 'id', 1)}",
                    )
                    p_ctx = PolicyContext(
                        capability=tc.capability,
                        action=tc.action,
                        session_id=turn.session_id,
                    )
                    pol_res = self.policy_engine.evaluate(tc, p_ctx)
                    turn.policy_result = pol_res

                    self._publish_event(
                        turn=turn,
                        event_type=CognitiveEventType.POLICY_DECIDED,
                        stage=CognitiveStage.POLICY,
                        summary=f"Policy decision: {pol_res.decision.value}",
                        metadata={"decision": pol_res.decision.value, "reason": pol_res.reason},
                    )

                    if pol_res.decision == PolicyDecision.DENY:
                        turn.status = TurnStatus.FAILED
                        turn.final_response = f"Policy denied execution: {pol_res.reason}"
                        policy_blocked = True
                        break
                    elif pol_res.decision in (PolicyDecision.ASK_PERMISSION, PolicyDecision.REQUIRE_CONFIRMATION):
                        turn.status = TurnStatus.WAITING_FOR_USER
                        turn.waiting_reason = pol_res.reason
                        turn.final_response = f"Action requires user confirmation: {pol_res.reason}"
                        policy_blocked = True
                        break

                if policy_blocked:
                    self._advance_stage(turn, CognitiveStage.RESPONSE)
                    if turn.status == TurnStatus.WAITING_FOR_USER:
                        self._publish_event(
                            turn=turn,
                            event_type=CognitiveEventType.TURN_WAITING,
                            stage=CognitiveStage.RESPONSE,
                            summary=turn.waiting_reason or "Waiting for user authorization",
                        )
                    else:
                        self._publish_event(
                            turn=turn,
                            event_type=CognitiveEventType.RESPONSE_COMPOSED,
                            stage=CognitiveStage.RESPONSE,
                            summary="Response composed after policy block",
                        )
                        self._advance_stage(turn, CognitiveStage.COMPLETED)
                    return self._build_result(turn, decision=decision, plan=plan)

            # -------------------------------------------------------------
            # STAGE 8 & 9: EXECUTION & OBSERVATION (Phase 2 & Phase 3.2)
            # -------------------------------------------------------------
            self._advance_stage(turn, CognitiveStage.EXECUTION)
            if self._check_deadline(turn, max_duration):
                return self._finalize_aborted(turn)

            def execute_fn(p: Plan) -> List[Result]:
                self._inject_context(p, request, history_entries, is_empty)
                results = self.execution_engine.execute(p)
                for r in results:
                    self._publish_event(
                        turn=turn,
                        event_type=CognitiveEventType.TOOL_EXECUTED,
                        stage=CognitiveStage.EXECUTION,
                        status="OK" if r.success else "FAILED",
                        summary=r.message or "Tool executed",
                        metadata={"success": r.success, "output_preview": str(r.output)[:100] if r.output else None},
                    )
                return results

            # -------------------------------------------------------------
            # STAGE 10 & 11: VERIFICATION & RECOVERY (Phase 2 & Phase 3.4)
            # -------------------------------------------------------------
            recovery_context = None
            if self.recovery_engine is not None:
                recovered_plan, results, verification, recovery_context = self.recovery_engine.recover(
                    original_goal=request.original_text,
                    initial_plan=plan,
                    execute_fn=execute_fn,
                    verify_fn=self.verifier.verify,
                )
                plan = recovered_plan
                turn.current_plan = plan
                turn.execution_results = results
                turn.verification_result = verification
                turn.recovery_result = recovery_context

                self._advance_stage(turn, CognitiveStage.OBSERVATION)
                self._publish_event(
                    turn=turn,
                    event_type=CognitiveEventType.OBSERVATION_RECEIVED,
                    stage=CognitiveStage.OBSERVATION,
                    summary=f"Observed {len(results)} execution results",
                )

                self._advance_stage(turn, CognitiveStage.VERIFICATION)
                self._publish_event(
                    turn=turn,
                    event_type=CognitiveEventType.VERIFICATION_COMPLETED,
                    stage=CognitiveStage.VERIFICATION,
                    summary=f"Verification completed: verified={getattr(verification, 'verified', True)}",
                    metadata={"verified": getattr(verification, 'verified', True)},
                )

                history = getattr(recovery_context, "plan_history", None) or getattr(recovery_context, "recovery_history", None)
                if recovery_context and history:
                    self._advance_stage(turn, CognitiveStage.RECOVERY)
                    last_entry = history[-1] if history else None
                    action = getattr(last_entry, "recovery_action", None) or getattr(recovery_context, "final_action", RecoveryAction.CONTINUE)
                    self._publish_event(
                        turn=turn,
                        event_type=CognitiveEventType.RECOVERY_STARTED,
                        stage=CognitiveStage.RECOVERY,
                        summary=f"Recovery triggered with action {action.value}",
                        metadata={"action": action.value},
                    )

                    if action == RecoveryAction.ASK_USER:
                        turn.status = TurnStatus.WAITING_FOR_USER
                        turn.waiting_reason = getattr(recovery_context, "failure_reason", None) or getattr(recovery_context, "last_error", None) or "Recovery required user clarification"
                        turn.final_response = turn.waiting_reason
                    elif action == RecoveryAction.ABORT:
                        turn.status = TurnStatus.ABORTED
                        turn.error = getattr(recovery_context, "failure_reason", None) or getattr(recovery_context, "last_error", None) or "Recovery aborted execution"

            else:
                results = execute_fn(plan)
                turn.execution_results = results

                self._advance_stage(turn, CognitiveStage.OBSERVATION)
                self._publish_event(
                    turn=turn,
                    event_type=CognitiveEventType.OBSERVATION_RECEIVED,
                    stage=CognitiveStage.OBSERVATION,
                    summary=f"Observed {len(results)} execution results",
                )

                self._advance_stage(turn, CognitiveStage.VERIFICATION)
                verification = self.verifier.verify(plan, results)
                turn.verification_result = verification
                self._publish_event(
                    turn=turn,
                    event_type=CognitiveEventType.VERIFICATION_COMPLETED,
                    stage=CognitiveStage.VERIFICATION,
                    summary=f"Verification completed: verified={getattr(verification, 'verified', True)}",
                    metadata={"verified": getattr(verification, 'verified', True)},
                )

            # -------------------------------------------------------------
            # STAGE 12: MEMORY (Read & User Write)
            # -------------------------------------------------------------
            self._advance_stage(turn, CognitiveStage.MEMORY)
            if not is_empty and self.memory_service is not None and not user_message_persisted:
                self.memory_service.add_message(
                    session_id=turn.session_id,
                    role=MessageRole.USER,
                    content=request.original_text,
                )
                user_message_persisted = True
                self._publish_event(
                    turn=turn,
                    event_type=CognitiveEventType.MEMORY_UPDATED,
                    stage=CognitiveStage.MEMORY,
                    summary="User turn recorded in conversational memory",
                )

            # -------------------------------------------------------------
            # STAGE 13: RESPONSE
            # -------------------------------------------------------------
            self._advance_stage(turn, CognitiveStage.RESPONSE)
            if turn.final_response is None:
                response = self.composer.compose(
                    request,
                    decision,
                    plan,
                    results,
                    verification,
                )
                turn.final_response = response
            else:
                response = turn.final_response

            # Persist assistant response in memory exactly once
            if not is_empty and self.memory_service is not None and not assistant_message_persisted:
                self.memory_service.add_message(
                    session_id=turn.session_id,
                    role=MessageRole.ASSISTANT,
                    content=response,
                )
                assistant_message_persisted = True

            self._publish_event(
                turn=turn,
                event_type=CognitiveEventType.RESPONSE_COMPOSED,
                stage=CognitiveStage.RESPONSE,
                summary="Assistant response composed and persisted",
            )

            # -------------------------------------------------------------
            # FINAL STATUS & COMPLETION
            # -------------------------------------------------------------
            if turn.status == TurnStatus.WAITING_FOR_USER:
                self._publish_event(
                    turn=turn,
                    event_type=CognitiveEventType.TURN_WAITING,
                    stage=CognitiveStage.RESPONSE,
                    summary=turn.waiting_reason or "Waiting for user input",
                )
                return self._build_result(turn, decision=decision, plan=plan, results=results, verification=verification, recovery=recovery_context)

            elif turn.status == TurnStatus.ABORTED:
                self._advance_stage(turn, CognitiveStage.ABORTED)
                self._publish_event(
                    turn=turn,
                    event_type=CognitiveEventType.TURN_ABORTED,
                    stage=CognitiveStage.ABORTED,
                    summary=turn.error or "Turn aborted",
                )
                return self._build_result(turn, decision=decision, plan=plan, results=results, verification=verification, recovery=recovery_context)

            else:
                self._advance_stage(turn, CognitiveStage.COMPLETED)
                is_success = getattr(verification, "verified", True)
                turn.status = TurnStatus.SUCCEEDED if is_success else TurnStatus.FAILED
                self._publish_event(
                    turn=turn,
                    event_type=CognitiveEventType.TURN_COMPLETED,
                    stage=CognitiveStage.COMPLETED,
                    summary="Cognitive turn completed successfully" if is_success else "Cognitive turn completed with failure",
                )
                return self._build_result(turn, decision=decision, plan=plan, results=results, verification=verification, recovery=recovery_context)

        except Exception as ex:
            logger.error("Exception in cognitive runtime turn %s: %s", turn.turn_id, ex, exc_info=True)
            turn.status = TurnStatus.FAILED
            turn.error = str(ex)
            # Try to advance to FAILED stage if possible
            if turn.current_stage not in (CognitiveStage.FAILED, CognitiveStage.COMPLETED, CognitiveStage.ABORTED):
                try:
                    self._advance_stage(turn, CognitiveStage.FAILED, summary=f"Turn failed due to exception: {ex}")
                except Exception:
                    turn.current_stage = CognitiveStage.FAILED

            self._publish_event(
                turn=turn,
                event_type=CognitiveEventType.STAGE_FAILED,
                stage=turn.current_stage,
                status="FAILED",
                summary=f"Turn error: {ex}",
                metadata={"error_type": type(ex).__name__},
            )

            # Re-raise pre-execution ValueError so assistant.py fallback functions as designed
            if isinstance(ex, ValueError) and turn.current_stage in (CognitiveStage.PLANNING, CognitiveStage.UNDERSTANDING, CognitiveStage.DECISION):
                raise

            raise

    def run(self, input_data: Any, session_id: Optional[str] = None) -> str:
        """Execute turn and return only final response string."""
        return self.execute_turn(input_data, session_id=session_id).response

    def _finalize_aborted(self, turn: CognitiveTurn) -> CognitiveTurnResult:
        """Helper to cleanly transition and return an aborted turn."""
        if turn.current_stage != CognitiveStage.ABORTED:
            try:
                self._advance_stage(turn, CognitiveStage.ABORTED, summary=turn.error or "Turn deadline exceeded")
            except Exception:
                turn.current_stage = CognitiveStage.ABORTED
        self._publish_event(
            turn=turn,
            event_type=CognitiveEventType.TURN_ABORTED,
            stage=CognitiveStage.ABORTED,
            summary=turn.error or "Turn aborted",
        )
        return self._build_result(turn)

    def _build_result(
        self,
        turn: CognitiveTurn,
        decision: Optional[Decision] = None,
        plan: Optional[Plan] = None,
        results: Optional[List[Result]] = None,
        verification: Optional[VerificationResult] = None,
        recovery: Optional[Any] = None,
    ) -> CognitiveTurnResult:
        """Construct bounded CognitiveTrace and immutable CognitiveTurnResult."""
        turn.end_time = time.time()
        events = self.event_sink.get_events(turn.turn_id)
        trace = CognitiveTrace(
            turn_id=turn.turn_id,
            session_id=turn.session_id,
            events=tuple(events[:self.limits.max_events]),
            start_time=turn.start_time,
            end_time=turn.end_time,
            final_status=turn.status,
            max_events=self.limits.max_events,
        )
        return CognitiveTurnResult(
            turn_id=turn.turn_id,
            session_id=turn.session_id,
            status=turn.status,
            stage=turn.current_stage,
            response=turn.final_response or "",
            trace=trace,
            request=turn.original_request,
            decision=decision or getattr(turn, "decision", None),
            plan=plan or turn.current_plan,
            results=results if results is not None else turn.execution_results,
            verification=verification or turn.verification_result,
            recovery=recovery or turn.recovery_result,
            waiting_reason=turn.waiting_reason,
            metadata=dict(turn.metadata),
        )
