from abc import ABC, abstractmethod
from typing import Optional, Dict, Set, Any

from core.models.reasoning import (
    ReasoningRequest,
    ReasoningResponse,
    ActionProposal,
    ProposalValidationResult,
    ProviderMetadata,
)
from core.models.perception import VisualScene


class ReasoningProviderInterface(ABC):
    """
    Model-neutral reasoning provider contract.
    Decouples reasoning from any vendor or backend (Ollama, local, cloud, mock).

    CORE SAFETY PRINCIPLE:
    A ReasoningProvider can ONLY propose reasoning outcomes and ActionProposals.
    It has NO ACCESS to executors, operating system, tools, databases, or devices.
    It can NEVER execute actions directly.
    """

    @property
    @abstractmethod
    def metadata(self) -> ProviderMetadata:
        """Return the provider's capability metadata and profile."""
        pass

    @abstractmethod
    def reason(self, request: ReasoningRequest) -> ReasoningResponse:
        """
        Inspect structured request context and return a structured ReasoningResponse.

        Args:
            request: Bounded, structured context including goal, visual scene,
                     grounded targets, memory, and recovery state.

        Returns:
            Structured ReasoningResponse (ActionProposal, observation request,
            clarification, completion, or abort).
        """
        pass


class ActionProposalValidatorInterface(ABC):
    """
    Deterministic boundary validating that an ActionProposal is safe, structurally
    valid, fresh, and permitted before it can be converted into an executable ToolCall.
    """

    @abstractmethod
    def validate(
        self,
        proposal: ActionProposal,
        current_scene: Optional[VisualScene] = None,
        allowed_capabilities: Optional[Dict[str, Set[str]]] = None,
    ) -> ProposalValidationResult:
        """
        Validate an ActionProposal and, if valid, produce an executable ToolCall.
        """
        pass
