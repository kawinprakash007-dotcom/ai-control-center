from typing import List, Optional, Union, Dict, Any
import uuid

from core.interfaces.reasoning_interface import ReasoningProviderInterface
from core.models.reasoning import (
    ReasoningRequest,
    ReasoningResponse,
    ReasoningOutcome,
    ActionProposal,
    ProviderMetadata,
    ModelCapabilityType,
)


class MockReasoningProvider(ReasoningProviderInterface):
    """
    Deterministic reference mock provider for testing Phase 3.7.
    Never calls any external LLM, network, or OS.
    Can be scripted with a queue of predetermined responses or default behaviours.
    """

    def __init__(
        self,
        provider_id: str = "mock_deterministic_reasoner",
        supported_capabilities: Optional[List[ModelCapabilityType]] = None,
        responses: Optional[List[Union[ReasoningResponse, Exception, Dict[str, Any]]]] = None,
    ):
        self._provider_id = provider_id
        self._capabilities = tuple(
            supported_capabilities
            if supported_capabilities is not None
            else [
                ModelCapabilityType.TEXT,
                ModelCapabilityType.VISION,
                ModelCapabilityType.STRUCTURED_OUTPUT,
                ModelCapabilityType.TOOL_REASONING,
            ]
        )
        self._metadata = ProviderMetadata(
            provider_id=self._provider_id,
            version="1.0.0",
            supported_capabilities=self._capabilities,
            is_local=True,
            description="Deterministic mock provider for Phase 3.7 test suite.",
        )
        self._response_queue: List[Union[ReasoningResponse, Exception, Dict[str, Any]]] = (
            list(responses) if responses else []
        )
        self.received_requests: List[ReasoningRequest] = []

    @property
    def metadata(self) -> ProviderMetadata:
        return self._metadata

    def enqueue_response(self, response: Union[ReasoningResponse, Exception, Dict[str, Any]]) -> None:
        self._response_queue.append(response)

    def reason(self, request: ReasoningRequest) -> ReasoningResponse:
        self.received_requests.append(request)

        if self._response_queue:
            next_resp = self._response_queue.pop(0)
            if isinstance(next_resp, Exception):
                raise next_resp
            if isinstance(next_resp, ReasoningResponse):
                return next_resp
            if isinstance(next_resp, dict):
                # Allow building response from dict for malformed/dynamic testing
                return ReasoningResponse(
                    turn_id=next_resp.get("turn_id", f"turn_{uuid.uuid4().hex[:8]}"),
                    outcome=next_resp.get("outcome", ReasoningOutcome.PROPOSE_ACTION),
                    proposal=next_resp.get("proposal"),
                    clarification_prompt=next_resp.get("clarification_prompt"),
                    completion_summary=next_resp.get("completion_summary"),
                    observation_request_reason=next_resp.get("observation_request_reason"),
                    abort_reason=next_resp.get("abort_reason"),
                    confidence=next_resp.get("confidence", 1.0),
                    raw_output=next_resp.get("raw_output"),
                    metadata=next_resp.get("metadata", {}),
                )

        # Default fallback behavior: inspect request and return completion or proposal
        if request.grounded_candidates:
            target = request.grounded_candidates[0]
            proposal = ActionProposal(
                proposal_id=f"prop_{uuid.uuid4().hex[:8]}",
                goal_reference=request.goal,
                action_type="computer",
                capability="computer",
                action="click",
                parameters={"x": target.click_coordinate[0], "y": target.click_coordinate[1]},
                target_reference=target.target_id,
                rationale=f"Click target {target.target_id}",
                confidence=target.confidence,
                observation_reference=target.source_observation_id,
            )
            return ReasoningResponse(
                turn_id=f"turn_{request.turn_index}",
                outcome=ReasoningOutcome.PROPOSE_ACTION,
                proposal=proposal,
                confidence=target.confidence,
            )

        return ReasoningResponse(
            turn_id=f"turn_{request.turn_index}",
            outcome=ReasoningOutcome.REPORT_COMPLETION,
            completion_summary=f"Completed reasoning for goal: {request.goal}",
            confidence=1.0,
        )
