from abc import ABC, abstractmethod
from typing import Optional, Any
from core.models.research import AgentAction
from core.models.web import ResearchState, EvidenceSet, CitationSet


class ResearchReasonerInterface(ABC):
    """
    Model-neutral reasoning contract for the Autonomous Research Agent.
    Decouples reasoning decisions from any specific model or backend.

    Safety Principle:
    The REASONER proposes an AgentAction.
    The RUNTIME validates, bounds, and executes.
    """

    @abstractmethod
    def decide_next_action(self, state: ResearchState) -> AgentAction:
        """
        Inspect current research state and propose the next research action.

        Args:
            state: Snapshot of current research state including objective,
                   iteration, search/fetch counts, bounded evidence, and
                   prior observations.

        Returns:
            AgentAction specifying proposed action_type, parameters, and reason.
        """
        pass

    def synthesize(self, state: ResearchState, citations: CitationSet) -> Optional[str]:
        """
        Optional reasoning hook to synthesize gathered evidence into a final response.
        If omitted or returns None, the runtime falls back to injected synthesizer
        or deterministic synthesis.
        """
        return None


class ResearchSynthesizerInterface(ABC):
    """
    Abstract contract for synthesizing collected evidence and citations
    into a structured final response.
    """

    @abstractmethod
    def synthesize(
        self,
        objective: str,
        evidence: EvidenceSet,
        citations: CitationSet,
    ) -> str:
        """
        Generate answer text from verified evidence and citations.
        """
        pass
