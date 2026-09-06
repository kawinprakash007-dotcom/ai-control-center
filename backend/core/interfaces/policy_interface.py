from abc import ABC, abstractmethod
from typing import Optional
from core.models.tool_call import ToolCall
from core.models.policy import PolicyContext, PolicyResult


class PolicyEngineInterface(ABC):
    """
    Model-neutral contract for deterministic policy, permission, and safety evaluation.
    Enforces authorization between model proposal / planning and tool execution.
    """

    @abstractmethod
    def evaluate(
        self,
        tool_call: ToolCall,
        context: Optional[PolicyContext] = None,
    ) -> PolicyResult:
        """
        Evaluate a proposed ToolCall within an optional PolicyContext.

        Args:
            tool_call: Proposed capability and action invocation.
            context: Contextual evaluation metadata (autonomy level, session, risk, etc.).

        Returns:
            Deterministic PolicyResult (ALLOW, DENY, ASK_PERMISSION, REQUIRE_CONFIRMATION).
        """
        pass
