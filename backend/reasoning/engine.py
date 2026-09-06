from typing import Optional, Dict, Any, List, Tuple, Callable
import uuid

from core.models.reasoning import (
    ReasoningRequest,
    ReasoningResponse,
    ReasoningOutcome,
    ActionProposal,
    ProposalValidationResult,
    ReasoningLimits,
)
from core.models.perception import VisualScene, GroundedTarget
from core.models.tool_call import ToolCall
from core.models.result import Result
from core.models.verification import VerificationResult
from core.interfaces.reasoning_interface import (
    ReasoningProviderInterface,
    ActionProposalValidatorInterface,
)
from core.models.recovery import RecoveryContext, RecoveryAction
from reasoning.validator import ActionProposalValidator


class ReasoningLoopTermination(Exception):
    """Raised when a reasoning cycle terminates due to safety bounds or limits."""
    pass


class ReasoningEngine:
    """
    ATLAS Reasoning Engine.
    Coordinates reasoning cycles between ReasoningProvider, ActionProposalValidator,
    and the execution/recovery pipeline.

    Architectural Safeguards:
    1. Model is NOT the brain: It only proposes outcomes and ActionProposals.
    2. Model proposals != ToolCalls. Only validated proposals become executable ToolCalls.
    3. Loop bounds: strict limits on turns, invalid proposals, observation requests,
       and consecutive no-progress signatures.
    4. Anti-loop protection: detects identical (scene, proposal, failure) cycles.
    5. Preserves user goal throughout.
    """

    def __init__(
        self,
        provider: ReasoningProviderInterface,
        validator: Optional[ActionProposalValidatorInterface] = None,
        limits: Optional[ReasoningLimits] = None,
    ):
        self.provider = provider
        self.validator = validator or ActionProposalValidator()
        self.limits = limits or ReasoningLimits()

    def run_turn(
        self,
        goal: str,
        task_description: Optional[str] = None,
        history: Tuple[Dict[str, str], ...] = (),
        memory_context: Tuple[str, ...] = (),
        visual_scene: Optional[VisualScene] = None,
        grounded_candidates: Tuple[GroundedTarget, ...] = (),
        screenshot_ref: Optional[str] = None,
        execution_history: Tuple[Dict[str, Any], ...] = (),
        verification_result: Optional[VerificationResult] = None,
        recovery_context: Optional[RecoveryContext] = None,
        turn_index: int = 1,
    ) -> Tuple[ReasoningResponse, Optional[ProposalValidationResult]]:
        """
        Execute a single reasoning turn:
        1. Build bounded ReasoningRequest.
        2. Call model-neutral ReasoningProvider.
        3. Validate response and any ActionProposal deterministically.
        4. Return structured response and validation result.
        """
        req = ReasoningRequest(
            goal=goal,
            task_description=task_description,
            history=history,
            memory_context=memory_context,
            visual_scene=visual_scene,
            grounded_candidates=grounded_candidates,
            screenshot_ref=screenshot_ref,
            execution_history=execution_history,
            verification_result=verification_result,
            recovery_context=recovery_context,
            turn_index=turn_index,
        )

        try:
            resp = self.provider.reason(req)
        except Exception as e:
            # Malformed/crashed provider fails closed to ABORT
            resp = ReasoningResponse(
                turn_id=f"err_{uuid.uuid4().hex[:8]}",
                outcome=ReasoningOutcome.ABORT,
                abort_reason=f"Reasoning provider failed: {str(e)}",
                confidence=0.0,
            )

        val_result = None
        if resp.outcome == ReasoningOutcome.PROPOSE_ACTION and resp.proposal:
            val_result = self.validator.validate(
                proposal=resp.proposal,
                current_scene=visual_scene,
            )

        return resp, val_result

    def run_reasoning_loop(
        self,
        goal: str,
        execute_fn: Callable[[ToolCall], Result],
        observe_fn: Optional[Callable[[], Optional[VisualScene]]] = None,
        verify_fn: Optional[Callable[[Result], VerificationResult]] = None,
        initial_scene: Optional[VisualScene] = None,
        memory_context: Tuple[str, ...] = (),
    ) -> Dict[str, Any]:
        """
        Execute a bounded, governed reasoning cycle until completion, clarification,
        or termination limit.
        """
        turn = 0
        invalid_proposals_count = 0
        observation_requests_count = 0
        consecutive_no_progress = 0

        current_scene = initial_scene
        execution_history: List[Dict[str, Any]] = []
        recent_signatures: List[str] = []

        last_verification: Optional[VerificationResult] = None
        last_recovery_context: Optional[RecoveryContext] = None

        while turn < self.limits.max_reasoning_turns:
            turn += 1

            resp, val_result = self.run_turn(
                goal=goal,
                memory_context=memory_context,
                visual_scene=current_scene,
                execution_history=tuple(execution_history),
                verification_result=last_verification,
                recovery_context=last_recovery_context,
                turn_index=turn,
            )

            # 1. Completion outcome
            if resp.outcome == ReasoningOutcome.REPORT_COMPLETION:
                return {
                    "outcome": ReasoningOutcome.REPORT_COMPLETION,
                    "summary": resp.completion_summary or "Completed",
                    "turns": turn,
                    "execution_history": execution_history,
                }

            # 2. Clarification outcome
            if resp.outcome == ReasoningOutcome.ASK_CLARIFICATION:
                return {
                    "outcome": ReasoningOutcome.ASK_CLARIFICATION,
                    "prompt": resp.clarification_prompt or "Clarification required",
                    "turns": turn,
                    "execution_history": execution_history,
                }

            # 3. Abort outcome
            if resp.outcome == ReasoningOutcome.ABORT:
                return {
                    "outcome": ReasoningOutcome.ABORT,
                    "reason": resp.abort_reason or "Reasoning aborted",
                    "turns": turn,
                    "execution_history": execution_history,
                }

            # 4. Request observation outcome
            if resp.outcome == ReasoningOutcome.REQUEST_OBSERVATION:
                observation_requests_count += 1
                if observation_requests_count > self.limits.max_observation_requests:
                    return {
                        "outcome": ReasoningOutcome.ABORT,
                        "reason": f"Exceeded max observation requests budget ({self.limits.max_observation_requests})",
                        "turns": turn,
                        "execution_history": execution_history,
                    }
                if observe_fn is not None:
                    current_scene = observe_fn()
                continue

            # 5. Propose action outcome
            if resp.outcome == ReasoningOutcome.PROPOSE_ACTION:
                assert val_result is not None
                if not val_result.is_valid or val_result.tool_call is None:
                    invalid_proposals_count += 1
                    if invalid_proposals_count >= self.limits.max_invalid_proposals:
                        return {
                            "outcome": ReasoningOutcome.ABORT,
                            "reason": f"Exceeded max invalid proposals budget ({self.limits.max_invalid_proposals}): {val_result.primary_error}",
                            "turns": turn,
                            "execution_history": execution_history,
                        }
                    # Feed validation failure back to history
                    execution_history.append({
                        "turn": turn,
                        "proposal": resp.proposal.to_dict() if resp.proposal else {},
                        "validation_error": val_result.primary_error,
                        "stale_target": val_result.stale_target,
                    })
                    continue

                # Valid proposal -> executable ToolCall
                tool_call = val_result.tool_call

                # Deterministic signature to detect loops
                sig = f"{current_scene.source_observation_id if current_scene else 'none'}|{tool_call.capability}|{tool_call.action}|{str(sorted(tool_call.parameters.items()))}"
                if sig in recent_signatures:
                    consecutive_no_progress += 1
                    if consecutive_no_progress >= self.limits.max_consecutive_no_progress:
                        return {
                            "outcome": ReasoningOutcome.ABORT,
                            "reason": f"No-progress loop detected: identical action proposed {consecutive_no_progress + 1} times without state progression",
                            "turns": turn,
                            "execution_history": execution_history,
                        }
                else:
                    consecutive_no_progress = 0
                recent_signatures.append(sig)

                # Execute via governing pipeline (ToolOrchestrator + PolicyEngine)
                result = execute_fn(tool_call)

                verif = None
                if verify_fn:
                    verif = verify_fn(result)
                    last_verification = verif

                execution_history.append({
                    "turn": turn,
                    "tool_call": tool_call.to_dict(),
                    "result": result.to_dict() if hasattr(result, "to_dict") else str(result),
                    "verified": verif.success if verif else getattr(result, "success", True),
                })

                # If execution failed or target was stale, refresh observation if observe_fn provided
                if not getattr(result, "success", True) and observe_fn is not None:
                    current_scene = observe_fn()

        return {
            "outcome": ReasoningOutcome.ABORT,
            "reason": f"Exceeded max reasoning turns budget ({self.limits.max_reasoning_turns})",
            "turns": turn,
            "execution_history": execution_history,
        }
