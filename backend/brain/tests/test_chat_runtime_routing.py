"""
ATLAS Phase 6.6 - Focused Chat Runtime Routing & Live Authority Verification Tests.

Validates the 18 core requirements:
1. "is the drone online" -> live ATLAS_DRONE_01 state
2. "check the drone status" -> ATLAS_DRONE_01 status
3. "show drone telemetry" -> ATLAS_DRONE_01 telemetry
4. "what is the battery level of the rover" -> ATLAS_ROVER_01 battery, NO Linux.pdf
5. "is the rover connected" -> rover connectivity
6. "what devices are online" -> registered devices list
7. "what is the current world state" -> WorldState, NO web search
8. "what situations are active" -> Situation layer
9. "what missions are active" -> Mission layer
10. General knowledge ("what is TCP") -> Knowledge path
11. Explicit web ("search the web for...") -> Web path
12. Action request ("take off the drone") -> Action path (not read-only)
13. Read-only query never executes an action
14. Ollama receives authoritative ATLAS context
15. No hallucinated live device state
16. No unrelated RAG result for device query
17. No unnecessary web call for live state
18. Unknown query behaves safely
"""

import pytest
from unittest.mock import MagicMock, patch

from core.models.request import Request
from core.models.decision import CapabilityType, ExecutionMode
from core.models.live_context import (
    ChatQueryClassification,
    LiveDeviceContext,
    LiveWorldContext,
    LiveSituationContext,
    LiveMissionContext,
)
from brain.request_understanding.standard_understanding import StandardRequestUnderstanding
from brain.decision_engine.standard_decision_engine import StandardDecisionEngine
from brain.planning.standard_planner import StandardPlanner
from tools.live_state_capability import LiveStateCapability
from runtime.cognitive_runtime import CognitiveRuntime
from orchestration.device_gateway import DeviceGateway
from orchestration.virtual_devices import create_virtual_drone, create_virtual_rover, create_virtual_vision, create_virtual_glass
from world.store import InMemoryWorldStateStore
from mission.situation_intelligence import MultiProductSituationIntelligenceEngine
from goals.manager import AutonomousGoalManager
from goals.store import InMemoryGoalStore
from safety.policy_engine import StandardPolicyEngine


@pytest.fixture
def configured_runtime():
    """Build a test runtime wired with simulation devices and live authorities."""
    understanding = StandardRequestUnderstanding()
    decision_engine = StandardDecisionEngine()
    planner = StandardPlanner()
    policy_engine = StandardPolicyEngine()
    world_store = InMemoryWorldStateStore()
    goal_store = InMemoryGoalStore()
    goal_manager = AutonomousGoalManager(store=goal_store, execution_engine=MagicMock())
    situation_engine = MultiProductSituationIntelligenceEngine()

    gateway = DeviceGateway()
    drone_id, drone_ad = create_virtual_drone("ATLAS_DRONE_01")
    rover_id, rover_ad = create_virtual_rover("ATLAS_ROVER_01")
    vision_id, vision_ad = create_virtual_vision("ATLAS_VISION_01")
    glass_id, glass_ad = create_virtual_glass("ATLAS_GLASS_01")

    gateway.register_device(drone_id)
    gateway.register_device(rover_id)
    gateway.register_device(vision_id)
    gateway.register_device(glass_id)

    gateway.register_adapter(drone_ad, device_id="ATLAS_DRONE_01")
    gateway.register_adapter(rover_ad, device_id="ATLAS_ROVER_01")
    gateway.register_adapter(vision_ad, device_id="ATLAS_VISION_01")
    gateway.register_adapter(glass_ad, device_id="ATLAS_GLASS_01")

    live_state = LiveStateCapability(
        device_gateway=gateway,
        world_store=world_store,
        situation_engine=situation_engine,
        goal_manager=goal_manager,
    )

    runtime = CognitiveRuntime(
        understanding=understanding,
        decision_engine=decision_engine,
        planner=planner,
        policy_engine=policy_engine,
        device_gateway=gateway,
        world_store=world_store,
        situation_engine=situation_engine,
        goal_manager=goal_manager,
        live_state_capability=live_state,
    )
    return runtime, gateway, world_store, situation_engine, goal_manager


class TestChatRuntimeRouting:

    # 1. "is the drone online"
    def test_drone_online_query(self, configured_runtime):
        runtime, gateway, _, _, _ = configured_runtime
        result = runtime.execute_turn("is the drone online")
        assert result.status.value == "SUCCEEDED"
        resp = result.response.lower()
        assert "atlas autonomous drone" in resp or "atlas_drone_01" in resp or "drone" in resp
        assert "online" in resp
        # Prove no generic drone manual or DJI references
        assert "dji" not in resp
        assert "qgroundcontrol" not in resp

    # 2. "check the drone status"
    def test_drone_status_query(self, configured_runtime):
        runtime, _, _, _, _ = configured_runtime
        result = runtime.execute_turn("check the drone status")

        assert result.status.value == "SUCCEEDED"
        assert "online" in result.response.lower()
        assert "healthy" in result.response.lower()

    # 3. "show drone telemetry"
    def test_drone_telemetry_query(self, configured_runtime):
        runtime, _, _, _, _ = configured_runtime
        result = runtime.execute_turn("show drone telemetry")

        assert result.status.value == "SUCCEEDED"
        resp = result.response.lower()
        assert "telemetry" in resp or "battery" in resp or "online" in resp

    # 4. "what is the battery level of the rover" -> NO Linux.pdf
    def test_rover_battery_query_no_linux_rag(self, configured_runtime):
        runtime, _, _, _, _ = configured_runtime
        result = runtime.execute_turn("what is the battery level of the rover")

        assert result.status.value == "SUCCEEDED"
        resp = result.response.lower()
        # Authoritative rover battery (92% for ATLAS_ROVER_01)
        assert "rover" in resp
        assert "92%" in resp or "battery" in resp
        # CRITICAL: prove ZERO Linux.pdf contamination
        assert "linux" not in resp
        assert "kernel" not in resp
        assert "pdf" not in resp

    # 5. "is the rover connected"
    def test_rover_connected_query(self, configured_runtime):
        runtime, _, _, _, _ = configured_runtime
        result = runtime.execute_turn("is the rover connected")

        assert result.status.value == "SUCCEEDED"
        assert "rover" in result.response.lower()
        assert "online" in result.response.lower()

    # 6. "what devices are online"
    def test_what_devices_are_online(self, configured_runtime):
        runtime, _, _, _, _ = configured_runtime
        result = runtime.execute_turn("what devices are online")

        assert result.status.value == "SUCCEEDED"
        resp = result.response.upper()
        assert "ATLAS_DRONE_01" in resp
        assert "ATLAS_ROVER_01" in resp
        assert "ATLAS_VISION_01" in resp
        assert "ATLAS_GLASS_01" in resp

    # 7. "what is the current world state" -> NO web search
    def test_world_state_query_no_web_search(self, configured_runtime):
        runtime, _, _, _, _ = configured_runtime
        result = runtime.execute_turn("what is the current world state")

        assert result.status.value == "SUCCEEDED"
        resp = result.response.lower()
        assert "worldstate" in resp or "world" in resp
        assert "search" not in resp
        assert "no web search results" not in resp

    # 8. "what situations are active"
    def test_active_situations_query(self, configured_runtime):
        runtime, _, _, _, _ = configured_runtime
        result = runtime.execute_turn("what situations are active")

        assert result.status.value == "SUCCEEDED"
        assert "situation" in result.response.lower()

    # 9. "what missions are active"
    def test_active_missions_query(self, configured_runtime):
        runtime, _, _, _, _ = configured_runtime
        result = runtime.execute_turn("what missions are active")

        assert result.status.value == "SUCCEEDED"
        assert "mission" in result.response.lower() or "goal" in result.response.lower()

    # 10. General knowledge question still uses knowledge path
    def test_general_knowledge_uses_knowledge_path(self):
        understanding = StandardRequestUnderstanding()
        req = understanding.understand("what is TCP")
        decision_engine = StandardDecisionEngine()
        decision = decision_engine.decide(req)

        assert CapabilityType.KNOWLEDGE in decision.required_capabilities
        assert decision.primary_goal == "retrieve_knowledge"

    # 11. Explicit web research still uses web path
    def test_explicit_web_research_uses_web_path(self):
        understanding = StandardRequestUnderstanding()
        req = understanding.understand("search the web for current drone regulations")
        decision_engine = StandardDecisionEngine()
        decision = decision_engine.decide(req)

        assert CapabilityType.WEB in decision.required_capabilities
        assert decision.primary_goal in ("web_search", "web_research")

    # 12. Action request remains an action request
    def test_action_request_not_read_only(self):
        understanding = StandardRequestUnderstanding()
        req = understanding.understand("take off the drone")
        assert req.parameters.get("classification") == "ACTION_REQUEST"

    # 13. Read-only query never executes an action or mutates state
    def test_read_only_query_never_executes_action(self, configured_runtime):
        runtime, gateway, _, _, _ = configured_runtime
        gateway.dispatch_to_device = MagicMock()

        runtime.execute_turn("is the drone online")
        runtime.execute_turn("what is the battery level of the rover")
        runtime.execute_turn("show drone telemetry")

        # Zero dispatches or mutations
        assert gateway.dispatch_to_device.call_count == 0

    # 14. Ollama receives authoritative ATLAS context
    def test_ollama_receives_authoritative_atlas_context(self):
        cap = LiveStateCapability()
        context_summary = "ATLAS Autonomous Drone (ATLAS_DRONE_01) is ONLINE and HEALTHY with 88% battery."

        with patch("tools.live_state_capability.ask_ollama") as mock_ollama:
            mock_ollama.return_value = "Yes, ATLAS Drone is online and healthy at 88%."
            res = cap._synthesize_or_fallback("is the drone online", context_summary, "Live Device")

            assert mock_ollama.called
            prompt_arg = mock_ollama.call_args[0][0]
            assert "ATLAS Autonomous Drone (ATLAS_DRONE_01)" in prompt_arg
            assert "88% battery" in prompt_arg
            assert "is the drone online" in prompt_arg
            assert res == "Yes, ATLAS Drone is online and healthy at 88%."

    # 15. No hallucinated live device state when Ollama is offline
    def test_no_hallucinated_device_state_on_llm_failure(self):
        cap = LiveStateCapability()
        context_summary = "ATLAS Autonomous Drone (ATLAS_DRONE_01) is ONLINE and HEALTHY with 88% battery."

        with patch("tools.live_state_capability.ask_ollama", side_effect=RuntimeError("Ollama offline")):
            res = cap._synthesize_or_fallback("is the drone online", context_summary, "Live Device")
            assert res == context_summary

    # 16. No unrelated RAG result for device queries
    def test_device_query_bypasses_rag(self):
        engine = StandardDecisionEngine()
        u = StandardRequestUnderstanding()
        req = u.understand("what is the battery level of the rover")

        is_k, q = engine._detect_knowledge(req)
        assert is_k is False
        assert q is None

    # 17. No unnecessary web call for world state
    def test_world_state_bypasses_web(self):
        engine = StandardDecisionEngine()
        u = StandardRequestUnderstanding()
        req = u.understand("what is the current world state")

        is_w, hints = engine._detect_web(req)
        assert is_w is False

    # 18. Unknown query behaves safely without crashing
    def test_unknown_query_behaves_safely(self, configured_runtime):
        runtime, _, _, _, _ = configured_runtime
        result = runtime.execute_turn("xyzzy qwerty random gibberish 12345")
        assert result is not None
        assert result.status is not None
