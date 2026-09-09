"""
ATLAS Phase 6.4 — Multi-Product Situation & Mission Intelligence Interfaces.

Defines abstract contracts for:
1. MultiProductSituationIntelligenceInterface: Cross-product situation correlation & contradiction evaluation.
2. ProductRoleSelectorInterface: Product suitability and candidate ranking.
3. MissionPlannerInterface: Deterministic semantic mission planning & replanning.
4. MissionCoordinatorInterface: Active mission tracking, goal creation coordination, and evidence verification.

CRITICAL ARCHITECTURAL RULES:
1. NO SECOND BRAIN: Interfaces represent orchestration & semantic planning contracts.
2. NO DIRECT EXECUTION: Interfaces DO NOT execute tools, device commands, or invoke LLMs.
3. GOAL AUTHORITY PRESERVED: MissionCoordinator routes goal requests through AutonomousGoalManager.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.models.mission import (
    EntityCorrelation,
    Mission,
    MissionObjective,
    MultiProductSituation,
    ProductEvidence,
    SituationContradiction,
)
from core.models.orchestration import Situation


class MultiProductSituationIntelligenceInterface(ABC):
    """
    Contract for correlating multiple edge product situations, detecting contradictions,
    and synthesizing multi-product situation intelligence.
    """

    @abstractmethod
    def evaluate_situations(
        self,
        situations: Sequence[Situation],
        world_state: Optional[Any] = None,
        events: Optional[Sequence[Any]] = None,
        now: Optional[float] = None,
    ) -> Sequence[MultiProductSituation]:
        """Synthesize unified MultiProductSituation instances from underlying Situations."""
        raise NotImplementedError

    @abstractmethod
    def correlate_entities(
        self,
        situations: Sequence[Situation],
        now: Optional[float] = None,
    ) -> Sequence[EntityCorrelation]:
        """Correlate entities observed across different products."""
        raise NotImplementedError

    @abstractmethod
    def detect_contradictions(
        self,
        situations: Sequence[Situation],
        now: Optional[float] = None,
    ) -> Sequence[SituationContradiction]:
        """Detect and preserve conflicting evidence across situations."""
        raise NotImplementedError

    @abstractmethod
    def get_active_multi_situations(self) -> Sequence[MultiProductSituation]:
        """Retrieve all currently active multi-product situations."""
        raise NotImplementedError

    @abstractmethod
    def get_multi_situation(self, situation_id: str) -> Optional[MultiProductSituation]:
        """Retrieve a specific multi-product situation by ID."""
        raise NotImplementedError


class ProductRoleSelectorInterface(ABC):
    """
    Contract for evaluating edge product suitability and ranking candidates for mission objectives.
    """

    @abstractmethod
    def select_candidate_products(
        self,
        objective: MissionObjective,
        available_devices: Sequence[Any],
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Sequence[str]:
        """Return prioritized list of device/product IDs suitable for executing an objective."""
        raise NotImplementedError

    @abstractmethod
    def rank_products_for_capability(
        self,
        capability: str,
        available_devices: Sequence[Any],
    ) -> Sequence[Tuple[str, float]]:
        """Rank available devices by suitability score for a given capability."""
        raise NotImplementedError


class MissionPlannerInterface(ABC):
    """
    Contract for generating dependency-ordered tactical mission plans and replanning upon failures.
    """

    @abstractmethod
    def plan_mission(
        self,
        situation: MultiProductSituation,
        available_devices: Sequence[Any],
        constraints: Optional[Dict[str, Any]] = None,
        now: Optional[float] = None,
    ) -> Mission:
        """Construct a new multi-product Mission from a MultiProductSituation."""
        raise NotImplementedError

    @abstractmethod
    def replan_mission(
        self,
        mission: Mission,
        failed_objective_id: str,
        reason: str,
        available_devices: Sequence[Any],
        now: Optional[float] = None,
    ) -> Mission:
        """Produce an adapted Mission plan reassigning or substituting failed objectives."""
        raise NotImplementedError


class MissionCoordinatorInterface(ABC):
    """
    Contract for managing multi-product mission lifecycles, evaluating evidence-driven completion,
    and routing goal requests to AutonomousGoalManager.
    """

    @abstractmethod
    def create_mission(self, mission: Mission) -> Mission:
        """Register and begin tracking an active multi-product mission."""
        raise NotImplementedError

    @abstractmethod
    def get_mission(self, mission_id: str) -> Optional[Mission]:
        """Retrieve a mission by its unique ID."""
        raise NotImplementedError

    @abstractmethod
    def list_active_missions(self) -> Sequence[Mission]:
        """List all currently active (in-progress) missions."""
        raise NotImplementedError

    @abstractmethod
    def ingest_evidence(
        self,
        evidence: ProductEvidence,
        mission_id: Optional[str] = None,
        now: Optional[float] = None,
    ) -> Sequence[Mission]:
        """Evaluate incoming product evidence against active mission objectives."""
        raise NotImplementedError

    @abstractmethod
    def step_coordination(self, now: Optional[float] = None) -> Sequence[Mission]:
        """Advance mission progression, verify completion criteria, and coordinate goal requests."""
        raise NotImplementedError

    @abstractmethod
    def abort_mission(
        self,
        mission_id: str,
        reason: str = "",
        now: Optional[float] = None,
    ) -> Optional[Mission]:
        """Abort an active mission and cancel associated goals via AutonomousGoalManager."""
        raise NotImplementedError

    @abstractmethod
    def pause_mission(
        self,
        mission_id: str,
        reason: str = "",
        now: Optional[float] = None,
    ) -> Optional[Mission]:
        """Pause an active mission and its associated goals."""
        raise NotImplementedError

    @abstractmethod
    def resume_mission(
        self,
        mission_id: str,
        now: Optional[float] = None,
    ) -> Optional[Mission]:
        """Resume a paused mission and its associated goals."""
        raise NotImplementedError
