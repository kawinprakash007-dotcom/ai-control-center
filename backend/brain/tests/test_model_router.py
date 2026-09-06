import pytest
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock

from core.models.reasoning import (
    ModelCapabilityType,
    ReasoningOutcome,
    ReasoningRequest,
    ReasoningResponse,
    ActionProposal,
)
from core.models.model_router import (
    LocalityRequirement,
    CostClass,
    LatencyClass,
    PrivacyClass,
    ModelDescriptor,
    ModelRequirements,
    RoutingResult,
)
from core.models.perception import VisualScene, GroundedTarget, VisualElement, BoundingBox, ElementType, PerceptionSource
from core.models.computer import ScreenDimensions
from core.models.tool_call import ToolCall
from core.models.result import Result
from core.models.policy import AutonomyLevel
from safety.policy_engine import StandardPolicyEngine
from tools.tool_orchestrator import ToolOrchestrator
from tools.capability_registry import CapabilityRegistry
from computer.mock_backend import MockComputerBackend
from computer.computer_capability import ComputerCapability
from reasoning.mock_provider import MockReasoningProvider
from reasoning.engine import ReasoningEngine
from routing.registry import ModelProviderRegistry, DuplicateProviderError
from routing.router import StandardModelRouter
from routing.requirement_mapper import derive_requirements_from_request


# =====================================================================
# A. CAPABILITY MODELS & DESCRIPTORS
# =====================================================================

class TestModelDescriptorDomain:
    def test_descriptor_immutability(self):
        desc = ModelDescriptor(
            provider_id="local_ollama",
            model_id="qwen3:8b",
            display_name="Qwen 3 8B",
            capabilities=(ModelCapabilityType.TEXT, ModelCapabilityType.STRUCTURED_OUTPUT),
            context_window=32768,
            is_local=True,
        )
        assert desc.provider_id == "local_ollama"
        assert desc.model_id == "qwen3:8b"
        assert desc.has_capability(ModelCapabilityType.TEXT)
        assert desc.has_capability(ModelCapabilityType.STRUCTURED_OUTPUT)
        assert not desc.has_capability(ModelCapabilityType.VISION)

        with pytest.raises(FrozenInstanceError):
            desc.enabled = False  # type: ignore

    def test_descriptor_negative_context_window_rejected(self):
        with pytest.raises(ValueError, match="context_window must be non-negative"):
            ModelDescriptor(
                provider_id="p1",
                model_id="m1",
                display_name="Bad Context",
                context_window=-10,
            )


# =====================================================================
# B. REQUIREMENTS
# =====================================================================

class TestModelRequirements:
    def test_requirements_defaults(self):
        req = ModelRequirements()
        assert req.required_capabilities == (ModelCapabilityType.TEXT,)
        assert req.locality_requirement == LocalityRequirement.CLOUD_ALLOWED
        assert req.privacy_requirement == PrivacyClass.PUBLIC_ALLOWED
        assert not req.structured_output_required
        assert req.minimum_context_window == 0

    def test_requirements_immutability(self):
        req = ModelRequirements(required_capabilities=(ModelCapabilityType.VISION,))
        with pytest.raises(FrozenInstanceError):
            req.minimum_context_window = 1000  # type: ignore


# =====================================================================
# C. REGISTRY
# =====================================================================

class TestModelProviderRegistry:
    def setup_method(self):
        self.registry = ModelProviderRegistry()
        self.provider = MockReasoningProvider(provider_id="prov_1")
        self.desc = ModelDescriptor(
            provider_id="prov_1",
            model_id="model_alpha",
            display_name="Alpha",
            capabilities=(ModelCapabilityType.TEXT,),
        )

    def test_register_and_lookup(self):
        self.registry.register(self.desc, self.provider)
        assert self.registry.get_provider("prov_1") is self.provider
        assert self.registry.get_descriptor("prov_1", "model_alpha") == self.desc

    def test_duplicate_registration_rejected(self):
        self.registry.register(self.desc, self.provider)
        with pytest.raises(DuplicateProviderError):
            self.registry.register(self.desc, self.provider)

    def test_enable_and_disable(self):
        self.registry.register(self.desc, self.provider)
        assert len(self.registry.list_descriptors(enabled_only=True)) == 1

        self.registry.set_enabled("prov_1", False, model_id="model_alpha")
        assert len(self.registry.list_descriptors(enabled_only=True)) == 0
        assert len(self.registry.list_descriptors(enabled_only=False)) == 1

        # Re-enable
        self.registry.set_enabled("prov_1", True, model_id="model_alpha")
        assert len(self.registry.list_descriptors(enabled_only=True)) == 1

    def test_unregister(self):
        self.registry.register(self.desc, self.provider)
        assert self.registry.unregister("prov_1", "model_alpha")
        assert self.registry.get_provider("prov_1") is None
        assert self.registry.get_descriptor("prov_1", "model_alpha") is None


# =====================================================================
# D. ROUTING (DETERMINISM, HARD CONSTRAINTS, SOFT PREFERENCES)
# =====================================================================

class TestModelRouter:
    def setup_method(self):
        self.registry = ModelProviderRegistry()
        self.router = StandardModelRouter(registry=self.registry)

        # Provider 1: Local text model, priority 50
        self.p1 = MockReasoningProvider(provider_id="local_text")
        self.d1 = ModelDescriptor(
            provider_id="local_text",
            model_id="tiny_text",
            display_name="Tiny Text",
            capabilities=(ModelCapabilityType.TEXT,),
            context_window=4096,
            is_local=True,
            priority=50,
        )

        # Provider 2: Local multimodal model, priority 70
        self.p2 = MockReasoningProvider(provider_id="local_vision")
        self.d2 = ModelDescriptor(
            provider_id="local_vision",
            model_id="vision_model",
            display_name="Local Vision",
            capabilities=(
                ModelCapabilityType.TEXT,
                ModelCapabilityType.VISION,
                ModelCapabilityType.STRUCTURED_OUTPUT,
                ModelCapabilityType.TOOL_REASONING,
            ),
            context_window=16384,
            is_local=True,
            priority=70,
        )

        # Provider 3: Cloud high-capacity model, priority 90
        self.p3 = MockReasoningProvider(provider_id="cloud_general")
        self.d3 = ModelDescriptor(
            provider_id="cloud_general",
            model_id="cloud_model",
            display_name="Cloud High Cap",
            capabilities=(
                ModelCapabilityType.TEXT,
                ModelCapabilityType.VISION,
                ModelCapabilityType.STRUCTURED_OUTPUT,
                ModelCapabilityType.TOOL_REASONING,
            ),
            context_window=131072,
            is_local=False,
            cost_class=CostClass.HIGH,
            priority=90,
        )

        self.registry.register(self.d1, self.p1)
        self.registry.register(self.d2, self.p2)
        self.registry.register(self.d3, self.p3)

    def test_deterministic_routing_with_priority(self):
        # Text only, cloud allowed -> cloud_general has priority 90 vs 70 vs 50
        req = ModelRequirements(required_capabilities=(ModelCapabilityType.TEXT,))
        res1 = self.router.route(req)
        res2 = self.router.route(req)
        assert res1.success
        assert res1.provider_id == "cloud_general"
        assert res1.model_id == "cloud_model"
        # Determinism guarantee
        assert res1 == res2

    def test_hard_locality_requirement_enforced(self):
        # Required LOCAL_ONLY -> cloud_general disqualified despite higher priority
        req = ModelRequirements(
            required_capabilities=(ModelCapabilityType.TEXT,),
            locality_requirement=LocalityRequirement.LOCAL_ONLY,
        )
        res = self.router.route(req)
        assert res.success
        assert res.provider_id == "local_vision"  # priority 70 > 50
        assert res.descriptor.is_local

    def test_hard_capability_requirement_enforced_no_silent_downgrade(self):
        # Required VISION -> local_text disqualified because it lacks VISION
        req = ModelRequirements(
            required_capabilities=(ModelCapabilityType.VISION,),
            locality_requirement=LocalityRequirement.LOCAL_ONLY,
        )
        res = self.router.route(req)
        assert res.success
        assert res.provider_id == "local_vision"
        assert res.descriptor.has_capability(ModelCapabilityType.VISION)

    def test_context_window_requirement_filtering(self):
        # Required minimum context 32000 -> only cloud_general (131072) qualifies
        req = ModelRequirements(
            required_capabilities=(ModelCapabilityType.TEXT,),
            minimum_context_window=32000,
        )
        res = self.router.route(req)
        assert res.success
        assert res.provider_id == "cloud_general"

        # Required context 200,000 -> none qualifies
        req_oversized = ModelRequirements(
            required_capabilities=(ModelCapabilityType.TEXT,),
            minimum_context_window=200000,
        )
        res_fail = self.router.route(req_oversized)
        assert not res_fail.success
        assert "Context window too small" in res_fail.error_reason

    def test_disabled_provider_is_never_selected(self):
        # Disable cloud_general
        self.registry.set_enabled("cloud_general", False, model_id="cloud_model")
        req = ModelRequirements(required_capabilities=(ModelCapabilityType.TEXT,))
        res = self.router.route(req)
        assert res.success
        assert res.provider_id == "local_vision"  # fallback to next highest priority

    def test_no_compatible_provider_returns_structured_failure(self):
        empty_router = StandardModelRouter(registry=ModelProviderRegistry())
        res = empty_router.route(ModelRequirements())
        assert not res.success
        assert "No registered reasoning providers" in res.error_reason

    def test_multiple_models_under_single_provider(self):
        provider = MockReasoningProvider(provider_id="multi_prov")
        m_small = ModelDescriptor(
            provider_id="multi_prov",
            model_id="fast_chat",
            display_name="Fast Chat",
            capabilities=(ModelCapabilityType.TEXT,),
            priority=40,
        )
        m_vision = ModelDescriptor(
            provider_id="multi_prov",
            model_id="expert_vision",
            display_name="Expert Vision",
            capabilities=(ModelCapabilityType.TEXT, ModelCapabilityType.VISION),
            priority=85,
        )
        reg = ModelProviderRegistry()
        reg.register(m_small, provider)
        reg.register(m_vision, provider)

        router = StandardModelRouter(registry=reg)

        # Query requiring vision
        res = router.route(ModelRequirements(required_capabilities=(ModelCapabilityType.VISION,)))
        assert res.success
        assert res.provider_id == "multi_prov"
        assert res.model_id == "expert_vision"


# =====================================================================
# E. REQUIREMENT MAPPER
# =====================================================================

class TestRequirementMapper:
    def test_mapper_visual_request(self):
        scene = VisualScene(
            source_observation_id="obs_1",
            screen_dimensions=ScreenDimensions(1920, 1080),
        )
        req = ReasoningRequest(goal="Click the button", visual_scene=scene)
        model_req = derive_requirements_from_request(req)
        assert ModelCapabilityType.VISION in model_req.required_capabilities
        assert model_req.structured_output_required

    def test_mapper_private_memory_request(self):
        req = ReasoningRequest(
            goal="Process secret key",
            memory_context=("User private secret",),
        )
        model_req = derive_requirements_from_request(req)
        assert model_req.locality_requirement == LocalityRequirement.LOCAL_ONLY
        assert model_req.privacy_requirement == PrivacyClass.STRICT_LOCAL


# =====================================================================
# F. REASONING ENGINE INTEGRATION WITH MODEL ROUTER
# =====================================================================

class TestReasoningEngineRouterIntegration:
    def test_engine_dynamic_routing_to_provider(self):
        reg = ModelProviderRegistry()
        provider = MockReasoningProvider(provider_id="routed_provider")
        desc = ModelDescriptor(
            provider_id="routed_provider",
            model_id="m_test",
            display_name="Routed Model",
            capabilities=(ModelCapabilityType.TEXT, ModelCapabilityType.STRUCTURED_OUTPUT),
            priority=100,
        )
        reg.register(desc, provider)
        router = StandardModelRouter(registry=reg)

        # Script provider response
        provider.enqueue_response(ReasoningResponse(
            turn_id="t1",
            outcome=ReasoningOutcome.REPORT_COMPLETION,
            completion_summary="Router successfully dispatched",
        ))

        engine = ReasoningEngine(router=router)
        loop_res = engine.run_reasoning_loop(
            goal="Test routing",
            execute_fn=lambda tc: Result.ok(message="ok"),
        )
        assert loop_res["outcome"] == ReasoningOutcome.REPORT_COMPLETION
        assert loop_res["summary"] == "Router successfully dispatched"
        assert len(provider.received_requests) == 1

    def test_engine_fails_closed_when_router_finds_no_provider(self):
        empty_router = StandardModelRouter(registry=ModelProviderRegistry())
        engine = ReasoningEngine(router=empty_router)
        loop_res = engine.run_reasoning_loop(
            goal="Impossible task without providers",
            execute_fn=lambda tc: Result.ok(message="ok"),
        )
        assert loop_res["outcome"] == ReasoningOutcome.ABORT
        assert "Model Router failed to select provider" in loop_res["reason"]


# =====================================================================
# G. SECURITY TESTS: ROUTER CANNOT EXECUTE TOOLS OR BYPASS POLICY
# =====================================================================

class TestRouterSecurityBoundaries:
    def test_router_has_no_execution_or_tool_access(self):
        router = StandardModelRouter()
        assert not hasattr(router, "execute")
        assert not hasattr(router, "executor")
        assert not hasattr(router, "backend")
        assert not hasattr(router, "orchestrator")
        assert not hasattr(router, "policy_engine")

    def test_model_proposal_from_routed_provider_still_passes_full_safety_stack(self):
        """
        Prove: Even when a provider is selected dynamically by ModelRouter,
        ActionProposalValidator and PolicyEngine remain authoritative over execution.
        """
        # 1. Setup Governed Stack
        backend = MockComputerBackend(screen_dimensions=ScreenDimensions(1920, 1080))
        cap = ComputerCapability(backend=backend)
        cap_reg = CapabilityRegistry()
        cap_reg.register("computer", cap)
        policy_engine = StandardPolicyEngine()
        orchestrator = ToolOrchestrator(registry=cap_reg, policy_engine=policy_engine)

        # 2. Setup Router Stack
        model_reg = ModelProviderRegistry()
        provider = MockReasoningProvider(provider_id="vision_prov")
        desc = ModelDescriptor(
            provider_id="vision_prov",
            model_id="v_model",
            display_name="Vision Provider",
            capabilities=(
                ModelCapabilityType.TEXT,
                ModelCapabilityType.VISION,
                ModelCapabilityType.STRUCTURED_OUTPUT,
                ModelCapabilityType.TOOL_REASONING,
            ),
        )
        model_reg.register(desc, provider)
        router = StandardModelRouter(registry=model_reg)

        # 3. Provider proposes valid click action
        prop = ActionProposal(
            proposal_id="p_click",
            goal_reference="Click icon",
            action_type="computer",
            capability="computer",
            action="click",
            parameters={"x": 200, "y": 400},
            confidence=0.9,
        )
        provider.enqueue_response(ReasoningResponse(
            turn_id="t1", outcome=ReasoningOutcome.PROPOSE_ACTION, proposal=prop
        ))
        provider.enqueue_response(ReasoningResponse(
            turn_id="t2", outcome=ReasoningOutcome.REPORT_COMPLETION, completion_summary="Finished"
        ))

        # 4. Execute via ReasoningEngine with router
        engine = ReasoningEngine(router=router)
        res = engine.run_reasoning_loop(
            goal="Click icon",
            execute_fn=lambda tc: orchestrator.execute(tc),
        )

        assert res["outcome"] == ReasoningOutcome.REPORT_COMPLETION
        assert len(backend.history) == 1
        assert backend.history[0]["action"] == "click"
        assert backend.history[0]["x"] == 200
        assert backend.history[0]["y"] == 400
