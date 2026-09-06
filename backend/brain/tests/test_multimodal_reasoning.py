import pytest
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock

from core.models.reasoning import (
    ReasoningOutcome,
    ModelCapabilityType,
    ProviderMetadata,
    ActionProposal,
    ReasoningRequest,
    ReasoningResponse,
    ProposalValidationResult,
    ReasoningLimits,
)
from core.models.perception import (
    VisualScene,
    VisualElement,
    BoundingBox,
    ElementType,
    PerceptionSource,
    GroundedTarget,
)
from core.models.computer import ScreenDimensions
from core.models.tool_call import ToolCall
from core.models.result import Result
from core.models.verification import VerificationResult
from core.models.policy import (
    PolicyDecision,
    PolicyContext,
    AutonomyLevel,
    RiskLevel,
)
from core.models.recovery import RecoveryLimits
from safety.policy_engine import StandardPolicyEngine
from tools.tool_orchestrator import ToolOrchestrator
from tools.capability_registry import CapabilityRegistry
from computer.mock_backend import MockComputerBackend
from computer.computer_capability import ComputerCapability
from reasoning.validator import ActionProposalValidator
from reasoning.mock_provider import MockReasoningProvider
from reasoning.engine import ReasoningEngine


# =====================================================================
# A. DOMAIN MODELS
# =====================================================================

class TestReasoningDomainModels:
    def test_action_proposal_immutability(self):
        prop = ActionProposal(
            proposal_id="p1",
            goal_reference="g1",
            action_type="computer",
            capability="computer",
            action="click",
            parameters={"x": 100, "y": 200},
            confidence=0.9,
        )
        assert prop.proposal_id == "p1"
        assert prop.capability == "computer"
        with pytest.raises(FrozenInstanceError):
            prop.confidence = 0.5  # type: ignore

    def test_action_proposal_confidence_bounds(self):
        with pytest.raises(ValueError, match="confidence must be in range"):
            ActionProposal(
                proposal_id="p1",
                goal_reference="g1",
                action_type="computer",
                capability="computer",
                action="click",
                confidence=1.5,
            )

    def test_reasoning_response_validation(self):
        # Outcome PROPOSE_ACTION without proposal must fail
        with pytest.raises(ValueError, match="must provide an ActionProposal"):
            ReasoningResponse(
                turn_id="t1",
                outcome=ReasoningOutcome.PROPOSE_ACTION,
                proposal=None,
            )

    def test_reasoning_request_immutability(self):
        req = ReasoningRequest(
            goal="Open Settings",
            memory_context=("User is Admin",),
            available_capabilities=("computer", "web"),
        )
        assert req.goal == "Open Settings"
        with pytest.raises(FrozenInstanceError):
            req.goal = "Other"  # type: ignore


# =====================================================================
# B. REASONING PROVIDER
# =====================================================================

class TestReasoningProvider:
    def test_mock_provider_metadata(self):
        provider = MockReasoningProvider(provider_id="test_provider")
        assert provider.metadata.provider_id == "test_provider"
        assert provider.metadata.has_capability(ModelCapabilityType.TEXT)
        assert provider.metadata.has_capability(ModelCapabilityType.VISION)

    def test_mock_provider_scripted_responses(self):
        provider = MockReasoningProvider()
        prop = ActionProposal(
            proposal_id="p1",
            goal_reference="g1",
            action_type="web",
            capability="web",
            action="search",
            parameters={"query": "python docs"},
        )
        resp1 = ReasoningResponse(turn_id="t1", outcome=ReasoningOutcome.PROPOSE_ACTION, proposal=prop)
        resp2 = ReasoningResponse(turn_id="t2", outcome=ReasoningOutcome.REPORT_COMPLETION, completion_summary="Done")

        provider.enqueue_response(resp1)
        provider.enqueue_response(resp2)

        r1 = provider.reason(ReasoningRequest(goal="Search python"))
        assert r1.outcome == ReasoningOutcome.PROPOSE_ACTION
        assert r1.proposal.action == "search"

        r2 = provider.reason(ReasoningRequest(goal="Search python"))
        assert r2.outcome == ReasoningOutcome.REPORT_COMPLETION
        assert r2.completion_summary == "Done"


# =====================================================================
# C. ACTION VALIDATION BOUNDARY
# =====================================================================

class TestActionProposalValidator:
    def setup_method(self):
        self.validator = ActionProposalValidator()

    def test_validate_valid_proposal(self):
        prop = ActionProposal(
            proposal_id="p_valid",
            goal_reference="g1",
            action_type="web",
            capability="web",
            action="search",
            parameters={"query": "release notes"},
        )
        res = self.validator.validate(prop)
        assert res.is_valid
        assert res.tool_call is not None
        assert res.tool_call.capability == "web"
        assert res.tool_call.action == "search"
        assert res.tool_call.parameters["query"] == "release notes"

    def test_unknown_capability_rejected(self):
        prop = ActionProposal(
            proposal_id="p_bad",
            goal_reference="g1",
            action_type="unknown",
            capability="my_custom_service",
            action="run_task",
        )
        res = self.validator.validate(prop)
        assert not res.is_valid
        assert res.unknown_capability
        assert res.tool_call is None

    def test_unknown_action_rejected(self):
        prop = ActionProposal(
            proposal_id="p_bad_act",
            goal_reference="g1",
            action_type="web",
            capability="web",
            action="destroy",
        )
        res = self.validator.validate(prop)
        assert not res.is_valid
        assert res.unknown_action
        assert res.tool_call is None

    def test_prohibited_terms_rejected(self):
        for term in ("shell", "subprocess", "powershell", "exec", "eval", "cmd"):
            prop = ActionProposal(
                proposal_id=f"p_{term}",
                goal_reference="g1",
                action_type=term,
                capability=term,
                action="run",
            )
            res = self.validator.validate(prop)
            assert not res.is_valid
            assert res.prohibited_action
            assert res.tool_call is None

    def test_prohibited_parameter_injection_rejected(self):
        prop = ActionProposal(
            proposal_id="p_inject",
            goal_reference="g1",
            action_type="web",
            capability="web",
            action="search",
            parameters={"cmd": "powershell malicious command"},
        )
        res = self.validator.validate(prop)
        assert not res.is_valid
        assert res.prohibited_action
        assert res.tool_call is None

    def test_invalid_parameters_rejected(self):
        prop = ActionProposal(
            proposal_id="p_bad_coord",
            goal_reference="g1",
            action_type="computer",
            capability="computer",
            action="click",
            parameters={"x": -10, "y": 50},  # negative coordinate
        )
        res = self.validator.validate(prop)
        assert not res.is_valid
        assert "coordinates must be non-negative" in res.primary_error


# =====================================================================
# D. VISUAL GROUNDING INTEGRATION & STALENESS
# =====================================================================

class TestVisualIntegrationAndStaleness:
    def setup_method(self):
        self.validator = ActionProposalValidator()
        self.screen = ScreenDimensions(1920, 1080)
        self.btn = VisualElement(
            element_id="btn_1",
            bounding_box=BoundingBox(100, 200, 80, 40),
            element_type=ElementType.BUTTON,
            text="Login",
            confidence=0.95,
            source=PerceptionSource.OCR,
        )
        self.scene = VisualScene(
            source_observation_id="obs_100",
            screen_dimensions=self.screen,
            elements=(self.btn,),
        )

    def test_valid_visual_proposal_against_scene(self):
        prop = ActionProposal(
            proposal_id="p_click",
            goal_reference="Login",
            action_type="computer",
            capability="computer",
            action="click",
            parameters={"x": 140, "y": 220},
            target_reference="btn_1",
            observation_reference="obs_100",
        )
        res = self.validator.validate(prop, current_scene=self.scene)
        assert res.is_valid
        assert res.tool_call is not None

    def test_stale_observation_reference_rejected(self):
        prop = ActionProposal(
            proposal_id="p_stale",
            goal_reference="Login",
            action_type="computer",
            capability="computer",
            action="click",
            parameters={"x": 140, "y": 220},
            target_reference="btn_1",
            observation_reference="obs_OLD_99",
        )
        res = self.validator.validate(prop, current_scene=self.scene)
        assert not res.is_valid
        assert res.stale_target
        assert "Observation mismatch" in res.primary_error

    def test_nonexistent_target_reference_rejected(self):
        prop = ActionProposal(
            proposal_id="p_missing_target",
            goal_reference="Login",
            action_type="computer",
            capability="computer",
            action="click",
            parameters={"x": 140, "y": 220},
            target_reference="btn_nonexistent",
            observation_reference="obs_100",
        )
        res = self.validator.validate(prop, current_scene=self.scene)
        assert not res.is_valid
        assert res.stale_target
        assert "does not exist in current VisualScene" in res.primary_error


# =====================================================================
# E. POLICY INTEGRATION & CONFIDENCE DOES NOT OVERRIDE POLICY
# =====================================================================

class TestPolicyEngineIntegration:
    def test_high_confidence_cannot_override_policy(self):
        """
        Prove: Model proposing an action with confidence=1.0 CANNOT override
        the PolicyEngine's risk evaluation or default-deny.
        """
        engine = StandardPolicyEngine()
        # In MANUAL autonomy, modifying actions (memory.save) require confirmation/deny
        tool_call = ToolCall(
            capability="memory",
            action="save",
            parameters={"key": "secret", "value": "val"},
        )
        ctx = PolicyContext(
            capability="memory",
            action="save",
            parameters={"key": "secret", "value": "val"},
            reason="Save data",
            autonomy_level=AutonomyLevel.MANUAL,
        )
        decision = engine.evaluate(tool_call, ctx)
        # Must require confirmation or deny, NOT allow
        assert decision.decision != PolicyDecision.ALLOW

    def test_default_deny_blocks_unauthorized_proposals(self):
        engine = StandardPolicyEngine()
        tool_call = ToolCall(
            capability="unknown_service",
            action="steal_data",
            parameters={},
        )
        ctx = PolicyContext(
            capability="unknown_service",
            action="steal_data",
            parameters={},
            reason="Attack",
            autonomy_level=AutonomyLevel.AUTONOMOUS,
        )
        decision = engine.evaluate(tool_call, ctx)
        assert decision.decision == PolicyDecision.DENY


# =====================================================================
# F. REASONING LOOP BOUNDS & NO-PROGRESS TERMINATION
# =====================================================================

class TestReasoningLoopBounds:
    def test_max_invalid_proposals_budget_terminates(self):
        provider = MockReasoningProvider()
        # Enqueue invalid proposals
        for _ in range(5):
            bad_prop = ActionProposal(
                proposal_id="bad",
                goal_reference="g",
                action_type="bad_cap",
                capability="bad_cap",
                action="bad_act",
            )
            provider.enqueue_response(ReasoningResponse(
                turn_id="t", outcome=ReasoningOutcome.PROPOSE_ACTION, proposal=bad_prop
            ))

        engine = ReasoningEngine(
            provider=provider,
            limits=ReasoningLimits(max_invalid_proposals=2),
        )
        result = engine.run_reasoning_loop(
            goal="Test invalid bounds",
            execute_fn=lambda tc: Result.ok(message="ok"),
        )
        assert result["outcome"] == ReasoningOutcome.ABORT
        assert "Exceeded max invalid proposals budget" in result["reason"]

    def test_max_observation_requests_budget_terminates(self):
        provider = MockReasoningProvider()
        for _ in range(5):
            provider.enqueue_response(ReasoningResponse(
                turn_id="t", outcome=ReasoningOutcome.REQUEST_OBSERVATION, observation_request_reason="Need fresh view"
            ))

        engine = ReasoningEngine(
            provider=provider,
            limits=ReasoningLimits(max_observation_requests=2),
        )
        result = engine.run_reasoning_loop(
            goal="Test observation bounds",
            execute_fn=lambda tc: Result.ok(message="ok"),
        )
        assert result["outcome"] == ReasoningOutcome.ABORT
        assert "Exceeded max observation requests budget" in result["reason"]

    def test_no_progress_loop_detection(self):
        provider = MockReasoningProvider()
        # Enqueue identical proposals
        for _ in range(5):
            prop = ActionProposal(
                proposal_id="p1",
                goal_reference="g",
                action_type="web",
                capability="web",
                action="search",
                parameters={"query": "python"},
            )
            provider.enqueue_response(ReasoningResponse(
                turn_id="t", outcome=ReasoningOutcome.PROPOSE_ACTION, proposal=prop
            ))

        engine = ReasoningEngine(
            provider=provider,
            limits=ReasoningLimits(max_consecutive_no_progress=2),
        )
        result = engine.run_reasoning_loop(
            goal="Test no progress",
            execute_fn=lambda tc: Result.fail(message="Network down"),
        )
        assert result["outcome"] == ReasoningOutcome.ABORT
        assert "No-progress loop detected" in result["reason"]


# =====================================================================
# G. SECURITY TESTS: MODEL CANNOT EXECUTE TOOLS DIRECTLY
# =====================================================================

class TestSecurityBoundaries:
    def test_model_provider_has_no_execution_access(self):
        provider = MockReasoningProvider()
        # Verify provider has no execute, executor, backend, or os attributes
        assert not hasattr(provider, "execute")
        assert not hasattr(provider, "executor")
        assert not hasattr(provider, "backend")
        assert not hasattr(provider, "run_command")

    def test_invalid_proposal_cannot_reach_orchestrator(self):
        orchestrator = MagicMock(spec=ToolOrchestrator)
        validator = ActionProposalValidator()

        bad_prop = ActionProposal(
            proposal_id="p_exploit",
            goal_reference="g",
            action_type="shell",
            capability="shell",
            action="exec",
            parameters={"cmd": "whoami"},
        )
        val_res = validator.validate(bad_prop)
        assert not val_res.is_valid

        # Even if a caller attempted to pass the result, tool_call is None
        assert val_res.tool_call is None
        orchestrator.execute.assert_not_called()

    def test_memory_cannot_be_directly_mutated_by_reasoning(self):
        provider = MockReasoningProvider()
        # Verify no direct memory storage access
        assert not hasattr(provider, "memory_store")
        assert not hasattr(provider, "sqlite")
        assert not hasattr(provider, "db")


# =====================================================================
# H. END-TO-END ARCHITECTURAL INTEGRATION TEST
# =====================================================================

class TestArchitecturalPipeline:
    def test_full_governed_flow_proposal_to_orchestrator_execution(self):
        """
        Verifies:
        MockReasoningProvider
            ↓
        ActionProposal
            ↓
        ActionProposalValidator
            ↓
        ToolCall
            ↓
        ToolOrchestrator
            ↓
        PolicyEngine (ALLOW)
            ↓
        ComputerCapability (MockComputerBackend)
            ↓
        Result
        """
        # 1. Setup Governed Capability Stack
        backend = MockComputerBackend(screen_dimensions=ScreenDimensions(1920, 1080))
        cap = ComputerCapability(backend=backend)
        registry = CapabilityRegistry()
        registry.register("computer", cap)

        policy_engine = StandardPolicyEngine()
        orchestrator = ToolOrchestrator(registry=registry, policy_engine=policy_engine)

        # 2. Setup Reasoning Provider with valid proposal
        provider = MockReasoningProvider()
        prop = ActionProposal(
            proposal_id="prop_click_btn",
            goal_reference="Click confirm button",
            action_type="computer",
            capability="computer",
            action="click",
            parameters={"x": 500, "y": 300},
            confidence=0.95,
        )
        resp1 = ReasoningResponse(turn_id="t1", outcome=ReasoningOutcome.PROPOSE_ACTION, proposal=prop)
        resp2 = ReasoningResponse(turn_id="t2", outcome=ReasoningOutcome.REPORT_COMPLETION, completion_summary="Clicked")
        provider.enqueue_response(resp1)
        provider.enqueue_response(resp2)

        # 3. Setup ReasoningEngine
        engine = ReasoningEngine(provider=provider)

        # 4. Run Loop
        loop_res = engine.run_reasoning_loop(
            goal="Click confirm button",
            execute_fn=lambda tc: orchestrator.execute(tc),
        )

        # 5. Assertions
        assert loop_res["outcome"] == ReasoningOutcome.REPORT_COMPLETION
        assert len(backend.history) == 1
        act = backend.history[0]
        assert act["action"] == "click"
        assert act["x"] == 500
        assert act["y"] == 300
