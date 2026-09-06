from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.models.anticipation import (
    Anticipation,
    AnticipationStatus,
    AnticipatoryDecision,
    EvidenceItem,
    TimeHorizon,
)


class EvidenceEvaluatorInterface(ABC):
    """
    Contract for evaluating supporting evidence for an anticipation.
    Separates confidence, freshness, and relevance into orthogonal metrics.
    Operates deterministically without direct tool or model execution.
    """

    @abstractmethod
    def evaluate_evidence(
        self,
        items: Sequence[EvidenceItem],
        active_goals: Sequence[Any],
        now: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        """
        Evaluate evidence items to produce:
        (confidence, freshness, relevance) scores, each in [0.0, 1.0].
        """
        pass


class InvalidationEngineInterface(ABC):
    """
    Contract for determining if an active anticipation hypothesis has become
    INVALIDATED, EXPIRED, or remains ACTIVE based on fresh facts.
    """

    @abstractmethod
    def evaluate_invalidation(
        self,
        anticipation: Anticipation,
        current_world_state: Optional[Any],
        active_goals: Sequence[Any],
        now: float,
    ) -> Tuple[AnticipationStatus, str]:
        """
        Evaluate whether an anticipation remains valid.

        Returns:
            (new_status, reason_explanation)
        """
        pass


class AnticipatoryAnalyzerInterface(ABC):
    """
    Contract for generating bounded candidate anticipations from existing facts.
    Operates on WorldState, active goals, recent events, and evidence.
    Does NOT write hypotheses into WorldState.
    """

    @abstractmethod
    def analyze(
        self,
        world_state: Optional[Any],
        active_goals: Sequence[Any],
        recent_events: Sequence[Any],
        evidence: Sequence[EvidenceItem],
        horizon_limit: Optional[TimeHorizon] = None,
        now: Optional[float] = None,
    ) -> List[Anticipation]:
        """
        Analyze current state, goals, events, and evidence to produce candidate anticipations.
        """
        pass


class AnticipatoryPlanningCoordinatorInterface(ABC):
    """
    Central coordinator contract for Phase 4.6 Anticipatory Planning.
    Coordinates analysis passes, evidence evaluation, invalidation checks,
    policy engine enforcement, and goal management routing.
    """

    @abstractmethod
    def evaluate_cycle(
        self,
        max_anticipations: int = 20,
    ) -> List[AnticipatoryDecision]:
        """
        Run a bounded, single-pass anticipatory evaluation cycle.
        """
        pass

    @abstractmethod
    def process_anticipation(
        self,
        anticipation: Anticipation,
    ) -> AnticipatoryDecision:
        """
        Process a single candidate anticipation hypothesis through invalidation,
        duplicate checks, policy engine, and AutonomousGoalManager.
        """
        pass

    @abstractmethod
    def get_active_anticipations(self) -> List[Anticipation]:
        """
        Return currently active, non-expired, non-invalidated anticipations.
        """
        pass

    @abstractmethod
    def get_decision_history(self, limit: int = 100) -> List[AnticipatoryDecision]:
        """
        Return history of past anticipatory decisions.
        """
        pass
