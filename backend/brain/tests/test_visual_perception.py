import time
import pytest
from dataclasses import FrozenInstanceError
from PIL import Image, ImageDraw

from core.models.computer import (
    ScreenDimensions,
    ComputerObservation,
)
from core.models.perception import (
    ElementType,
    PerceptionSource,
    SpatialRelation,
    BoundingBox,
    VisualElement,
    VisualScene,
    GroundingRequest,
    GroundedTarget,
    GroundingResult,
    GroundingAmbiguity,
)
from core.models.tool_call import ToolCall
from core.models.policy import PolicyContext, AutonomyLevel, PolicyDecision
from core.models.plan import Plan
from core.models.task import Task
from core.models.result import Result
from core.models.recovery import RecoveryLimits, ExecutionOutcome
from core.models.verification import VerificationResult
from safety.policy_engine import StandardPolicyEngine
from tools.tool_orchestrator import ToolOrchestrator
from tools.capability_registry import CapabilityRegistry
from computer.mock_backend import MockComputerBackend
from computer.computer_capability import ComputerCapability
from perception.deduplicator import deduplicate_elements
from perception.accessibility_provider import AccessibilityPerceptionProvider
from perception.ocr_provider import OCRPerceptionProvider
from perception.cv_provider import CVPerceptionProvider
from perception.mock_provider import MockPerceptionProvider
from perception.engine import VisualPerceptionEngine
from perception.grounder import TargetGrounder
from brain.execution.standard_execution_engine import StandardExecutionEngine
from brain.verification.standard_verifier import StandardVerifier
from brain.recovery.recovery_engine import StandardRecoveryEngine


# ============================================================================
# A. DOMAIN MODELS
# ============================================================================

class TestPerceptionDomainModels:
    def test_bounding_box_valid_and_calculations(self):
        box = BoundingBox(x=10, y=20, width=100, height=50)
        assert box.x == 10
        assert box.y == 20
        assert box.width == 100
        assert box.height == 50
        assert box.center == (60, 45)
        assert box.area == 5000
        assert box.right == 110
        assert box.bottom == 70

        # Point containment
        assert box.contains_point(60, 45) is True
        assert box.contains_point(10, 20) is True
        assert box.contains_point(110, 70) is True
        assert box.contains_point(5, 5) is False

    def test_bounding_box_invalid_raises(self):
        with pytest.raises(ValueError):
            BoundingBox(x=-1, y=0, width=10, height=10)
        with pytest.raises(ValueError):
            BoundingBox(x=0, y=-5, width=10, height=10)
        with pytest.raises(ValueError):
            BoundingBox(x=0, y=0, width=-10, height=10)

    def test_bounding_box_iou_and_intersects(self):
        box1 = BoundingBox(x=0, y=0, width=100, height=100)
        box2 = BoundingBox(x=50, y=0, width=100, height=100)
        assert box1.intersects(box2) is True
        # intersection is 50x100 = 5000, union is 10000 + 10000 - 5000 = 15000. IoU = 5000 / 15000 = 0.333...
        assert abs(box1.iou(box2) - (1.0 / 3.0)) < 0.01

        box3 = BoundingBox(x=200, y=200, width=50, height=50)
        assert box1.intersects(box3) is False
        assert box1.iou(box3) == 0.0

    def test_bounding_box_spatial_relations(self):
        target = BoundingBox(x=100, y=100, width=50, height=50)
        below = BoundingBox(x=100, y=180, width=50, height=50)
        above = BoundingBox(x=100, y=20, width=50, height=50)
        right = BoundingBox(x=180, y=100, width=50, height=50)
        left = BoundingBox(x=20, y=100, width=50, height=50)

        assert below.relation_to(target) == SpatialRelation.BELOW
        assert above.relation_to(target) == SpatialRelation.ABOVE
        assert right.relation_to(target) == SpatialRelation.RIGHT_OF
        assert left.relation_to(target) == SpatialRelation.LEFT_OF

    def test_visual_element_and_immutability(self):
        box = BoundingBox(x=10, y=10, width=100, height=30)
        el = VisualElement(
            element_id="btn_1",
            element_type=ElementType.BUTTON,
            bounding_box=box,
            text="Submit Form",
            confidence=0.95,
            source=PerceptionSource.OCR,
        )
        assert el.center == (60, 25)
        assert el.matches_text("Submit") is True
        assert el.matches_text("submit form", exact=True) is True
        assert el.matches_text("Cancel") is False

        with pytest.raises(FrozenInstanceError):
            el.text = "Mutated"  # type: ignore

    def test_visual_element_confidence_bounds(self):
        box = BoundingBox(x=0, y=0, width=10, height=10)
        with pytest.raises(ValueError):
            VisualElement(element_id="el", element_type=ElementType.TEXT, bounding_box=box, confidence=1.5)
        with pytest.raises(ValueError):
            VisualElement(element_id="el", element_type=ElementType.TEXT, bounding_box=box, confidence=-0.1)

    def test_visual_scene_immutability(self):
        screen = ScreenDimensions(width=1920, height=1080)
        scene = VisualScene(
            source_observation_id="obs_001",
            screen_dimensions=screen,
            elements=(),
            timestamp=100.0,
        )
        assert scene.element_count == 0
        with pytest.raises(FrozenInstanceError):
            scene.timestamp = 200.0  # type: ignore


# ============================================================================
# B. OCR PROVIDER & FALLBACK
# ============================================================================

class TestOCRPerceptionProvider:
    def test_ocr_provider_availability(self):
        provider = OCRPerceptionProvider()
        # Should report boolean without throwing
        assert isinstance(provider.is_available(), bool)

    def test_ocr_fallback_when_no_image(self):
        provider = OCRPerceptionProvider()
        screen = ScreenDimensions(width=1920, height=1080)
        obs_empty = ComputerObservation(timestamp=10.0, screen_dimensions=screen)
        results = provider.perceive(obs_empty)
        assert results == []


# ============================================================================
# C. COMPUTER VISION PROVIDER & SYNTHETIC FIXTURE
# ============================================================================

class TestCVPerceptionProvider:
    def test_cv_provider_synthetic_fixture(self, tmp_path):
        # Create a synthetic image containing a button rectangle
        img = Image.new("RGB", (400, 300), color="white")
        draw = ImageDraw.Draw(img)
        draw.rectangle([(50, 50), (200, 100)], fill="black", outline="black")
        synthetic_path = str(tmp_path / "synthetic_ui.png")
        img.save(synthetic_path)

        provider = CVPerceptionProvider(min_area=50, min_width=20, min_height=15)
        if not provider.is_available():
            pytest.skip("OpenCV not available")

        screen = ScreenDimensions(width=400, height=300)
        obs = ComputerObservation(
            timestamp=1.0,
            screen_dimensions=screen,
            screenshot_path=synthetic_path,
        )
        elements = provider.perceive(obs)
        assert len(elements) >= 1
        found_box = elements[0].bounding_box
        # Check that bounding box approximately surrounds the rectangle
        assert abs(found_box.x - 50) <= 10
        assert abs(found_box.y - 50) <= 10


# ============================================================================
# D. ACCESSIBILITY PROVIDER
# ============================================================================

class TestAccessibilityPerceptionProvider:
    def test_accessibility_tree_parsing(self):
        screen = ScreenDimensions(width=1920, height=1080)
        ui_metadata = {
            "elements": [
                {
                    "id": "acc_btn_ok",
                    "role": "Button",
                    "name": "OK",
                    "bounds": [100, 200, 80, 30],
                    "is_enabled": True,
                },
                {
                    "id": "acc_txt_user",
                    "role": "Edit",
                    "name": "Username",
                    "bounds": [100, 150, 200, 30],
                },
            ]
        }
        obs = ComputerObservation(
            timestamp=1.0,
            screen_dimensions=screen,
            ui_tree_metadata=ui_metadata,
        )

        provider = AccessibilityPerceptionProvider()
        elements = provider.perceive(obs)
        assert len(elements) == 2

        btn = elements[0]
        assert btn.element_type == ElementType.BUTTON
        assert btn.text == "OK"
        assert btn.bounding_box.x == 100
        assert btn.bounding_box.y == 200
        assert btn.confidence == 1.0
        assert btn.source == PerceptionSource.ACCESSIBILITY

        edit = elements[1]
        assert edit.element_type == ElementType.INPUT
        assert edit.text == "Username"


# ============================================================================
# E. MULTI-SOURCE FUSION & DEDUPLICATION
# ============================================================================

class TestMultiSourceDeduplication:
    def test_ocr_and_accessibility_fusion(self):
        # Accessibility detected button at (100, 200, 80, 30)
        box_acc = BoundingBox(x=100, y=200, width=80, height=30)
        el_acc = VisualElement(
            element_id="acc_1",
            element_type=ElementType.BUTTON,
            bounding_box=box_acc,
            text="Save",
            confidence=0.90,
            source=PerceptionSource.ACCESSIBILITY,
        )

        # OCR detected text "Save" at slightly shifted (102, 201, 76, 28)
        box_ocr = BoundingBox(x=102, y=201, width=76, height=28)
        el_ocr = VisualElement(
            element_id="ocr_1",
            element_type=ElementType.TEXT,
            bounding_box=box_ocr,
            text="Save",
            confidence=0.85,
            source=PerceptionSource.OCR,
        )

        fused = deduplicate_elements([el_acc, el_ocr], iou_threshold=0.3)
        assert len(fused) == 1
        merged = fused[0]
        assert merged.element_type == ElementType.BUTTON
        assert merged.text == "Save"
        # Confidence boosted on multi-source agreement
        assert merged.confidence > 0.90
        assert "ocr" in merged.properties["sources"]
        assert "accessibility" in merged.properties["sources"]

    def test_distinct_elements_not_deduplicated(self):
        el1 = VisualElement("1", ElementType.BUTTON, BoundingBox(0, 0, 50, 50), "A")
        el2 = VisualElement("2", ElementType.BUTTON, BoundingBox(200, 200, 50, 50), "B")
        fused = deduplicate_elements([el1, el2])
        assert len(fused) == 2


# ============================================================================
# F. SCENE QUERIES
# ============================================================================

class TestSceneQueries:
    @pytest.fixture
    def sample_scene(self):
        screen = ScreenDimensions(1920, 1080)
        elements = [
            VisualElement("txt_1", ElementType.TEXT, BoundingBox(50, 50, 100, 30), "Login"),
            VisualElement("inp_1", ElementType.INPUT, BoundingBox(50, 90, 100, 35), ""),
            VisualElement("btn_1", ElementType.BUTTON, BoundingBox(50, 150, 100, 40), "Submit"),
            VisualElement("btn_2", ElementType.BUTTON, BoundingBox(160, 150, 100, 40), "Cancel"),
        ]
        return VisualScene(
            source_observation_id="obs_test",
            screen_dimensions=screen,
            elements=tuple(elements),
            timestamp=1000.0,
        )

    def test_find_text(self, sample_scene):
        res = sample_scene.find_text("Submit")
        assert len(res) == 1
        assert res[0].element_id == "btn_1"

    def test_find_by_type(self, sample_scene):
        buttons = sample_scene.find_by_type(ElementType.BUTTON)
        assert len(buttons) == 2
        assert {b.element_id for b in buttons} == {"btn_1", "btn_2"}

    def test_find_in_region(self, sample_scene):
        # Region covering (0,0) to (200, 100) includes txt_1 and inp_1
        region = BoundingBox(0, 0, 200, 100)
        found = sample_scene.find_in_region(region)
        assert len(found) == 2
        assert {e.element_id for e in found} == {"txt_1", "inp_1"}

    def test_spatial_relation_query(self, sample_scene):
        txt_login = sample_scene.get_element_by_id("txt_1")
        # Find elements below Login text
        below = sample_scene.find_by_relation(txt_login, SpatialRelation.BELOW)
        assert len(below) >= 1
        assert below[0].element_id == "inp_1"


# ============================================================================
# G. TARGET GROUNDING & AMBIGUITY
# ============================================================================

class TestTargetGrounding:
    def test_ground_exact_text_match(self):
        screen = ScreenDimensions(1920, 1080)
        btn = VisualElement("b1", ElementType.BUTTON, BoundingBox(200, 300, 100, 40), "Download Report")
        scene = VisualScene("obs_1", screen, (btn,), timestamp=time.time())

        grounder = TargetGrounder()
        req = GroundingRequest(text="Download Report")
        res = grounder.ground(req, scene)

        assert res.success is True
        assert res.target is not None
        assert res.target.element.element_id == "b1"
        assert res.target.click_coordinate == (250, 320)
        assert res.target.confidence >= 0.8

    def test_ground_with_spatial_relation(self):
        screen = ScreenDimensions(1920, 1080)
        label = VisualElement("lbl", ElementType.TEXT, BoundingBox(100, 100, 150, 30), "Password")
        field = VisualElement("txt", ElementType.INPUT, BoundingBox(100, 140, 200, 35), "")
        scene = VisualScene("obs_1", screen, (label, field), timestamp=time.time())

        grounder = TargetGrounder()
        req = GroundingRequest(
            element_type=ElementType.INPUT,
            relation=SpatialRelation.BELOW,
            relation_target_text="Password",
        )
        res = grounder.ground(req, scene)

        assert res.success is True
        assert res.target.element.element_id == "txt"

    def test_ambiguity_detection_when_multiple_matches_tie(self):
        screen = ScreenDimensions(1920, 1080)
        btn1 = VisualElement("btn_top", ElementType.BUTTON, BoundingBox(50, 100, 80, 30), "Edit", confidence=0.8)
        btn2 = VisualElement("btn_bottom", ElementType.BUTTON, BoundingBox(50, 300, 80, 30), "Edit", confidence=0.8)
        scene = VisualScene("obs_ambig", screen, (btn1, btn2), timestamp=time.time())

        grounder = TargetGrounder(ambiguity_margin=0.08)
        req = GroundingRequest(text="Edit", element_type=ElementType.BUTTON)
        res = grounder.ground(req, scene)

        assert res.success is False
        assert res.error_code == "AMBIGUOUS_TARGET"
        assert res.ambiguity is not None
        assert res.ambiguity.is_ambiguous is True
        assert res.ambiguity.candidate_count == 2

    def test_target_not_found(self):
        screen = ScreenDimensions(1920, 1080)
        btn = VisualElement("b1", ElementType.BUTTON, BoundingBox(50, 50, 100, 30), "Submit")
        scene = VisualScene("obs_1", screen, (btn,), timestamp=time.time())

        grounder = TargetGrounder()
        req = GroundingRequest(text="NonExistentButton")
        res = grounder.ground(req, scene)

        assert res.success is False
        assert res.error_code == "TARGET_NOT_FOUND"


# ============================================================================
# H. STALE TARGET PROTECTION
# ============================================================================

class TestStaleTargetProtection:
    def test_stale_target_rejected_on_observation_id_mismatch(self):
        screen = ScreenDimensions(1920, 1080)
        btn = VisualElement("b1", ElementType.BUTTON, BoundingBox(100, 100, 80, 30), "Submit")
        old_scene = VisualScene("obs_OLD_123", screen, (btn,), timestamp=time.time())

        grounder = TargetGrounder()
        req = GroundingRequest(text="Submit")

        # Current desktop observation has changed to obs_NEW_456
        current_obs = ComputerObservation(
            timestamp=time.time(),
            screen_dimensions=screen,
            observation_id="obs_NEW_456",
        )

        res = grounder.ground(req, old_scene, current_observation=current_obs)
        assert res.success is False
        assert res.error_code == "TARGET_STALE"
        assert "Re-observation required" in res.reason

    def test_grounded_target_is_stale_helper(self):
        box = BoundingBox(10, 10, 50, 20)
        el = VisualElement("el_1", ElementType.BUTTON, box, "OK")
        target = GroundedTarget(
            target_id="tgt_1",
            element=el,
            click_coordinate=(35, 20),
            confidence=0.9,
            source_observation_id="obs_100",
            observation_timestamp=time.time() - 40.0,
        )

        assert target.is_stale("obs_200") is True
        # Stale by age
        assert target.is_stale("obs_100", max_age_seconds=30.0, current_time=time.time()) is True
        # Fresh
        assert target.is_stale("obs_100", max_age_seconds=60.0, current_time=time.time()) is False


# ============================================================================
# I. LOOP & CAPACITY BOUNDS
# ============================================================================

class TestPerceptionLoopBounds:
    def test_max_elements_capacity_bound(self):
        screen = ScreenDimensions(1920, 1080)
        obs = ComputerObservation(timestamp=1.0, screen_dimensions=screen)

        # Create mock provider with 300 elements
        mock_prov = MockPerceptionProvider()
        for i in range(300):
            mock_prov.add_element(f"el_{i}", ElementType.TEXT, 10, i * 2, 50, 10, f"Text {i}")

        engine = VisualPerceptionEngine(providers=[mock_prov], max_elements=50)
        scene = engine.perceive(obs)

        assert scene.element_count <= 50
        assert scene.metadata["raw_detected_count"] == 300
        assert scene.metadata["fused_count"] == 50

    def test_max_passes_bound(self):
        screen = ScreenDimensions(1920, 1080)
        obs = ComputerObservation(timestamp=1.0, screen_dimensions=screen)

        p1 = MockPerceptionProvider()
        p2 = MockPerceptionProvider()
        p3 = MockPerceptionProvider()
        p4 = MockPerceptionProvider()

        engine = VisualPerceptionEngine(providers=[p1, p2, p3, p4], max_passes=2)
        scene = engine.perceive(obs)

        assert scene.metadata["passes"] == 2
        assert p1.call_count == 1
        assert p2.call_count == 1
        assert p3.call_count == 0
        assert p4.call_count == 0


# ============================================================================
# J. ORCHESTRATION & POLICY INTEGRATION
# ============================================================================

class TestGroundingOrchestratorIntegration:
    def test_grounded_target_converts_to_governed_tool_call(self):
        mock_backend = MockComputerBackend()
        cap = ComputerCapability(backend=mock_backend)
        registry = CapabilityRegistry(capabilities={"computer": cap})
        policy_engine = StandardPolicyEngine()
        orchestrator = ToolOrchestrator(registry=registry, policy_engine=policy_engine)

        screen = ScreenDimensions(1920, 1080)
        btn = VisualElement("b1", ElementType.BUTTON, BoundingBox(150, 250, 100, 40), "Sign In")
        scene = VisualScene("obs_1", screen, (btn,), timestamp=time.time())

        grounder = TargetGrounder()
        req = GroundingRequest(text="Sign In")
        res = grounder.ground(req, scene)

        assert res.success is True
        target = res.target

        # Convert grounded target to ToolCall parameters
        call_params = target.to_tool_call_params(action="click")
        tool_call = ToolCall(capability="computer", action="click", parameters=call_params)

        # Execute through standard governed ToolOrchestrator with policy
        ctx = PolicyContext(
            capability="computer",
            action="click",
            parameters=call_params,
            autonomy_level=AutonomyLevel.ASSISTED,
        )
        result = orchestrator.execute(tool_call, context=ctx)

        assert result.success is True
        assert mock_backend.click_count == 1
        assert mock_backend.cursor_pos == (200, 270)

    def test_policy_denies_grounded_click_in_manual_mode(self):
        mock_backend = MockComputerBackend()
        cap = ComputerCapability(backend=mock_backend)
        registry = CapabilityRegistry(capabilities={"computer": cap})
        policy_engine = StandardPolicyEngine()
        orchestrator = ToolOrchestrator(registry=registry, policy_engine=policy_engine)

        screen = ScreenDimensions(1920, 1080)
        btn = VisualElement("b1", ElementType.BUTTON, BoundingBox(100, 100, 50, 50), "Delete")
        scene = VisualScene("obs_1", screen, (btn,), timestamp=time.time())

        grounder = TargetGrounder()
        res = grounder.ground(GroundingRequest(text="Delete"), scene)
        target = res.target

        tool_call = ToolCall(capability="computer", action="click", parameters=target.to_tool_call_params())
        ctx_manual = PolicyContext(
            capability="computer",
            action="click",
            parameters=tool_call.parameters,
            autonomy_level=AutonomyLevel.MANUAL,
        )

        result = orchestrator.execute(tool_call, context=ctx_manual)
        # Denied due to manual confirmation required
        assert result.success is False
        assert "permission" in result.message.lower() or "denied" in result.message.lower()
        # Mock backend was NOT clicked
        assert mock_backend.click_count == 0


# ============================================================================
# K. ARCHITECTURAL SECURITY BOUNDARIES
# ============================================================================

class TestPerceptionSecurityBoundaries:
    def test_grounder_cannot_invoke_backend_directly(self):
        grounder = TargetGrounder()
        # Ensure no reference to ComputerBackend or execution method exists
        assert not hasattr(grounder, "backend")
        assert not hasattr(grounder, "execute")
        assert not hasattr(grounder, "click")

    def test_perception_engine_cannot_invoke_backend_directly(self):
        engine = VisualPerceptionEngine()
        assert not hasattr(engine, "backend")
        assert not hasattr(engine, "execute")
        assert not hasattr(engine, "click")


# ============================================================================
# L. VERIFICATION INTEGRATION
# ============================================================================

class TestVisualVerification:
    def test_visual_state_verification_succeeds(self):
        verifier = StandardVerifier()
        screen = ScreenDimensions(1920, 1080)
        success_el = VisualElement("txt_ok", ElementType.TEXT, BoundingBox(100, 100, 200, 30), "Welcome Alice")
        scene = VisualScene("obs_2", screen, (success_el,), timestamp=time.time())

        # Plan expected Welcome message
        plan = Plan(
            goal="login to portal",
            steps=[
                Task(id=1, type="computer", action="Click Login", tool="computer", status="completed")
            ],
            status="completed",
        )
        results = [
            Result.ok(message="Clicked login", data={"visual_scene": scene.to_dict()})
        ]

        ver_res = verifier.verify(plan, results)
        assert ver_res.verified is True
        assert ver_res.status == "verified"


# ============================================================================
# M. RECOVERY INTEGRATION
# ============================================================================

class TestVisualRecoveryIntegration:
    def test_stale_target_triggers_observation_recovery(self):
        """When target is stale, recovery loop refreshes observation and replans."""
        screen = ScreenDimensions(1920, 1080)
        mock_backend = MockComputerBackend()
        cap = ComputerCapability(backend=mock_backend)
        registry = CapabilityRegistry(capabilities={"computer": cap})
        orchestrator = ToolOrchestrator(registry=registry)

        attempt_count = 0
        def execute_fn(p: Plan) -> list:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count == 1:
                p.status = "failed"
                p.steps[0].status = "failed"
                return [Result.fail(message="Target is stale (503 transient): display updated")]
            p.status = "completed"
            p.steps[0].status = "completed"
            return [Result.ok(message="Re-observed fresh target and clicked successfully")]

        verifier = StandardVerifier()
        recovery_engine = StandardRecoveryEngine(limits=RecoveryLimits(max_attempts=3, max_retries_per_action=2))

        plan = Plan(
            goal="click submit button",
            steps=[
                Task(id=1, type="computer", action="Click Submit", tool="computer", parameters={"action": "click", "x": 10, "y": 10}, status="pending")
            ],
            status="pending",
        )

        final_plan, results, ver, ctx = recovery_engine.recover(
            original_goal="click submit button",
            initial_plan=plan,
            execute_fn=execute_fn,
            verify_fn=verifier.verify,
        )

        assert ver.verified is True
        assert ctx.attempt == 2
        assert ctx.outcome == ExecutionOutcome.SUCCESS

    def test_replanning_does_not_bypass_policy_on_grounded_action(self):
        """When replanning proposes a computer action, policy is still strictly enforced."""
        mock_backend = MockComputerBackend()
        cap = ComputerCapability(backend=mock_backend)
        registry = CapabilityRegistry(capabilities={"computer": cap})
        policy_engine = StandardPolicyEngine()
        orchestrator = ToolOrchestrator(registry=registry, policy_engine=policy_engine)

        def execute_with_policy(p: Plan) -> list:
            results = []
            for step in p.steps:
                tool_call = ToolCall(capability="computer", action="click", parameters={"x": 50, "y": 50})
                # In MANUAL autonomy, click requires permission -> denied without user consent
                ctx_manual = PolicyContext(
                    capability="computer",
                    action="click",
                    parameters={"x": 50, "y": 50},
                    autonomy_level=AutonomyLevel.MANUAL,
                )
                res = orchestrator.execute(tool_call, context=ctx_manual)
                if not res.success:
                    step.status = "failed"
                    results.append(res)
                    break
                step.status = "completed"
                results.append(res)
            return results

        verifier = StandardVerifier()
        recovery_engine = StandardRecoveryEngine(limits=RecoveryLimits(max_attempts=2))

        initial_plan = Plan(
            goal="click restricted target",
            steps=[Task(id=1, type="computer", action="Click", tool="computer", status="pending")],
            status="pending",
        )

        final_plan, results, ver, ctx = recovery_engine.recover(
            original_goal="click restricted target",
            initial_plan=initial_plan,
            execute_fn=execute_with_policy,
            verify_fn=verifier.verify,
        )

        # Policy denial prevents execution and cannot be bypassed by replanning
        assert ver.verified is False
        assert mock_backend.click_count == 0
