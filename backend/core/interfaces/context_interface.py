from abc import ABC, abstractmethod
from typing import Optional

from core.models.context import (
    CognitiveState,
    ContextBudget,
    ContextSelection,
)


class ContextManagerInterface(ABC):
    """
    Model-neutral interface for cognitive context management.
    Determines which available information should be presented to the
    reasoning layer for the CURRENT cognitive step.

    CORE ARCHITECTURAL INVARIANTS:
    1. ContextManager is NOT the brain.
    2. ContextManager does NOT store memory or mutate persistent state.
    3. ContextManager does NOT call LLMs or remote embedding services.
    4. ContextManager does NOT execute tools or make policy decisions.
    5. ContextManager does NOT select models or call ModelRouter.
    6. ContextManager produces bounded, structured ContextSelection.
    """

    @abstractmethod
    def build_context(
        self,
        cognitive_state: CognitiveState,
        budget: Optional[ContextBudget] = None,
    ) -> ContextSelection:
        """
        Build bounded, prioritized, and relevant context selection from the current cognitive state.

        Args:
            cognitive_state: Bounded snapshot of current goal, task, perception,
                             evidence, and execution/recovery state.
            budget: Optional custom budget constraints (max items, max tokens, item length).

        Returns:
            Structured ContextSelection containing selected and omitted items with audit metadata.
        """
        pass
