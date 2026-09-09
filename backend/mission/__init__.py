"""
ATLAS Phase 6.4 — Multi-Product Situation & Mission Intelligence Package.

Provides:
- MultiProductSituationIntelligenceEngine: Cross-product evidence correlation & contradiction evaluation.
- ProductRoleSelector: Evaluates device suitability and ranks candidates for objectives.
- MissionPlanner: Dependency-aware mission planning and bounded replanning.
- MissionCoordinator: Active mission lifecycle management, AutonomousGoalManager integration, and completion tracking.
- MissionTimeline: Bounded, chronological audit trail of mission events.
"""

from mission.coordinator import MissionCoordinator
from mission.planner import MissionPlanner
from mission.role_selector import ProductRoleSelector
from mission.situation_intelligence import MultiProductSituationIntelligenceEngine
from mission.timeline import MissionTimeline

__all__ = [
    "MultiProductSituationIntelligenceEngine",
    "ProductRoleSelector",
    "MissionPlanner",
    "MissionCoordinator",
    "MissionTimeline",
]
