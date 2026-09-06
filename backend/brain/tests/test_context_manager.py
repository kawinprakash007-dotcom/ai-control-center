import pytest
import time
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock

from core.models.context import (
    ContextSource,
    ContextPriority,
    SensitivityLevel,
    AttentionFocus,
    ContextItem,
    ContextBudget,
    ContextSelection,
    CognitiveState,
)
from core.models.result import Result
from core.models.verification import VerificationResult
from core.models.recovery import RecoveryContext, ExecutionOutcome, FailureClassification
from core.models.perception import VisualScene, GroundedTarget, VisualElement, BoundingBox, ElementType, PerceptionSource
from core.models.computer import ScreenDimensions
from core.models.web import EvidenceItem, EvidenceSet, ResearchState
from core.models.memory import ChatMessage, MemoryEntry, MessageRole
from core.models.reasoning import (
    ReasoningRequest,
    ReasoningResponse,
    ReasoningOutcome,
    ModelCapabilityType,
)
from core.models.model_router import (
    ModelDescriptor,
    ModelRequirements,
    LocalityRequirement,
    PrivacyClass,
)
from core.interfaces.context_interface import ContextManagerInterface
from context.context_manager import StandardContextManager
from context.relevance import (
    tokenize_deterministic,
    compute_token_overlap,
    compute_recency_score,
    score_candidate_relevance,
)
from context.policy import (
    apply_attention_focus_weights,
    deduplicate_items_deterministic,
    compact_and_budget_items,
)
from routing.registry import ModelProviderRegistry
from routing.router import StandardModelRouter
from routing.requirement_mapper import derive_requirements_from_request
from reasoning.mock_provider import MockReasoningProvider
from reasoning.engine import ReasoningEngine


# =====================================================================
# A. DOMAIN MODELS
# =====================================================================

class TestContextDomainModels:
    def test_context_item_validation_and_immutability(self):
        item = ContextItem(
            item_id="item_1",
            source=ContextSource.CURRENT_GOAL,
            content="Organize workspace files",
            relevance=0.9,
            priority=ContextPriority.HIGHEST,
            recency=1.0,
            confidence=1.0,
            is_protected=True,
        )
        assert item.item_id == "item_1"
        assert item.source == ContextSource.CURRENT_GOAL
        assert item.priority == ContextPriority.HIGHEST
        assert item.token_estimate > 0
        assert item.is_protected is True

        with pytest.raises(FrozenInstanceError):
            item.relevance = 0.5  # type: ignore

    def test_context_item_invalid_relevance_or_confidence_rejected(self):
        with pytest.raises(ValueError, match="relevance must be in range"):
            ContextItem(
                item_id="bad_rel",
                source=ContextSource.MEMORY,
                content="test",
                relevance=1.5,
            )

        with pytest.raises(ValueError, match="confidence must be in range"):
            ContextItem(
                item_id="bad_conf",
                source=ContextSource.MEMORY,
                content="test",
                confidence=-0.1,
            )

    def test_context_item_compaction(self):
        long_content = "Word " * 200  # 1000 chars
        item = ContextItem(
            item_id="long_item",
            source=ContextSource.KNOWLEDGE,
            content=long_content,
            relevance=0.8,
        )
        compacted = item.compact(max_chars=100)
        assert len(compacted.content) <= 125
        assert "[truncated]" in compacted.content
        assert compacted.metadata.get("compacted") is True
        assert compacted.metadata.get("original_length") == len(long_content)

    def test_cognitive_state_validation(self):
        with pytest.raises(ValueError, match="original_goal must be a non-empty string"):
            CognitiveState(original_goal="")

        state = CognitiveState(
            original_goal="Find invoice for March",
            current_task="Search Downloads directory",
            attention_focus=AttentionFocus.PLANNING,
        )
        assert state.original_goal == "Find invoice for March"
        assert state.current_task == "Search Downloads directory"
        assert state.attention_focus == AttentionFocus.PLANNING

        with pytest.raises(FrozenInstanceError):
            state.original_goal = "Changed"  # type: ignore

    def test_context_budget_validation(self):
        with pytest.raises(ValueError, match="max_items must be positive"):
            ContextBudget(max_items=0)

        with pytest.raises(ValueError, match="max_tokens must be positive"):
            ContextBudget(max_tokens=-10)

        budget = ContextBudget(max_items=15, max_tokens=2000, max_content_length_per_item=500)
        assert budget.max_items == 15
        assert budget.max_tokens == 2000

    def test_context_selection_methods(self):
        item1 = ContextItem(
            item_id="i1",
            source=ContextSource.CURRENT_GOAL,
            content="Goal text",
            is_protected=True,
        )
        item2 = ContextItem(
            item_id="i2",
            source=ContextSource.MEMORY,
            content="Secret api key",
            sensitivity=SensitivityLevel.SECRET,
        )
        selection = ContextSelection(
            selected_items=(item1, item2),
            total_candidates=2,
            selection_reason="All candidates fit budget.",
            attention_focus=AttentionFocus.GOAL,
        )
        assert selection.item_count == 2
        assert selection.has_sensitive_content() is True
        assert len(selection.get_items_by_source(ContextSource.CURRENT_GOAL)) == 1
        assert len(selection.get_items_by_source(ContextSource.MEMORY)) == 1
        assert len(selection.get_items_by_source(ContextSource.WEB_EVIDENCE)) == 0

        summary = selection.formatted_summary()
        assert "ContextSelection" in summary
        assert "Goal text" in summary


# =====================================================================
# B. SOURCE PRIORITY & PROTECTION
# =====================================================================

class TestSourcePriorityAndProtection:
    def test_current_goal_and_task_always_protected(self):
        cm = StandardContextManager()
        state = CognitiveState(
            original_goal="Delete old log files",
            current_task="Scan /var/log directory",
        )
        selection = cm.build_context(state)
        goal_items = selection.get_items_by_source(ContextSource.CURRENT_GOAL)
        task_items = selection.get_items_by_source(ContextSource.CURRENT_TASK)

        assert len(goal_items) == 1
        assert goal_items[0].is_protected is True
        assert goal_items[0].priority == ContextPriority.HIGHEST

        assert len(task_items) == 1
        assert task_items[0].is_protected is True
        assert task_items[0].priority == ContextPriority.HIGHEST

    def test_failed_verification_is_protected_blocker(self):
        cm = StandardContextManager()
        verif = VerificationResult(
            verified=False,
            status="failed",
            confidence=0.95,
            reason="Disk write permission denied",
            failed_task_id=3,
        )
        state = CognitiveState(
            original_goal="Save output report",
            verification_state=verif,
        )
        selection = cm.build_context(state)
        verif_items = selection.get_items_by_source(ContextSource.VERIFICATION)
        assert len(verif_items) == 1
        assert verif_items[0].is_protected is True
        assert verif_items[0].priority == ContextPriority.HIGHEST
        assert "Disk write permission denied" in verif_items[0].content

    def test_successful_verification_not_unduly_protected(self):
        cm = StandardContextManager()
        verif = VerificationResult(
            verified=True,
            status="verified",
            confidence=1.0,
            reason="All tasks passed",
        )
        state = CognitiveState(
            original_goal="Run checks",
            verification_state=verif,
        )
        selection = cm.build_context(state)
        verif_items = selection.get_items_by_source(ContextSource.VERIFICATION)
        assert len(verif_items) == 1
        assert verif_items[0].is_protected is False


# =====================================================================
# C. RELEVANCE SCORING
# =====================================================================

class TestRelevanceScoring:
    def test_direct_goal_match_scores_higher_than_unrelated(self):
        goal = "Fix postgres database connection timeout"
        task = "Update postgresql.conf"

        rel_match = score_candidate_relevance(
            content="Found postgresql timeout setting in postgresql.conf",
            goal=goal,
            task=task,
        )
        unrelated = score_candidate_relevance(
            content="Today the weather is sunny with clear skies",
            goal=goal,
            task=task,
        )
        assert rel_match > unrelated
        assert rel_match > 0.5
        assert unrelated < 0.2

    def test_task_match_boosts_relevance(self):
        goal = "Deploy web application"
        task = "Build frontend docker container"

        task_match = score_candidate_relevance(
            content="Docker build completed for frontend container",
            goal=goal,
            task=task,
        )
        generic_match = score_candidate_relevance(
            content="General Linux package manager apt-get update",
            goal=goal,
            task=task,
        )
        assert task_match > generic_match


# =====================================================================
# D. RECENCY HANDLING
# =====================================================================

class TestRecencyHandling:
    def test_recent_item_outranks_older_equivalent_item(self):
        now = time.time()
        goal = "Resolve memory pressure"

        recent_item = ContextItem(
            item_id="rec_recent",
            source=ContextSource.EXECUTION_RESULT,
            content="Memory usage at 92% detected",
            relevance=0.8,
            priority=ContextPriority.HIGH,
            recency=0.95,
            confidence=1.0,
            timestamp=now - 5,
        )
        older_item = ContextItem(
            item_id="rec_older",
            source=ContextSource.EXECUTION_RESULT,
            content="Memory usage at 92% detected earlier",
            relevance=0.8,
            priority=ContextPriority.HIGH,
            recency=0.20,
            confidence=1.0,
            timestamp=now - 7200,
        )

        score_recent = apply_attention_focus_weights(recent_item, AttentionFocus.EXECUTION)
        score_older = apply_attention_focus_weights(older_item, AttentionFocus.EXECUTION)
        assert score_recent >= score_older

    def test_recency_cannot_override_strong_relevance(self):
        goal = "Compile C++ kernel module"
        now = time.time()

        # Older, highly relevant item
        older_relevant_rel = score_candidate_relevance(
            content="Kernel module compile error: missing header linux/init.h",
            goal=goal,
            recency_score=0.2,
            confidence=1.0,
        )

        # Recent, completely irrelevant item
        recent_irrelevant_rel = score_candidate_relevance(
            content="Chocolate cake recipe: 2 cups flour 1 cup sugar",
            goal=goal,
            recency_score=1.0,
            confidence=1.0,
        )

        assert older_relevant_rel > recent_irrelevant_rel


# =====================================================================
# E. CONFIDENCE & PROVENANCE
# =====================================================================

class TestConfidenceAndProvenance:
    def test_higher_confidence_preferred_when_equivalent(self):
        item_high_conf = ContextItem(
            item_id="c_high",
            source=ContextSource.WEB_EVIDENCE,
            content="Official documentation confirms Python 3.13 support",
            relevance=0.8,
            confidence=1.0,
        )
        item_low_conf = ContextItem(
            item_id="c_low",
            source=ContextSource.WEB_EVIDENCE,
            content="Forum user speculation says Python 3.13 might work",
            relevance=0.8,
            confidence=0.4,
        )
        score_high = apply_attention_focus_weights(item_high_conf, AttentionFocus.RESEARCH)
        score_low = apply_attention_focus_weights(item_low_conf, AttentionFocus.RESEARCH)
        assert score_high > score_low

    def test_provenance_metadata_preserved_in_selection(self):
        cm = StandardContextManager()
        ev = EvidenceItem(
            id="ev-12345",
            title="Postgres Config Guide",
            url="https://postgresql.org/docs/config.html",
            domain="postgresql.org",
            content="Detailed explanation of timeout parameters.",
            retrieved_at="2026-09-06T12:00:00Z",
        )
        state = CognitiveState(
            original_goal="Configure database timeout",
            web_evidence=(ev,),
        )
        selection = cm.build_context(state)
        ev_items = selection.get_items_by_source(ContextSource.WEB_EVIDENCE)
        assert len(ev_items) == 1
        assert ev_items[0].metadata["domain"] == "postgresql.org"
        assert ev_items[0].metadata["url"] == "https://postgresql.org/docs/config.html"


# =====================================================================
# F. BUDGET & CRITICAL PROTECTION
# =====================================================================

class TestBudgetAndCompaction:
    def test_max_item_budget_enforced(self):
        cm = StandardContextManager()
        budget = ContextBudget(max_items=3, max_tokens=1000)

        knowledge_list = [f"Supporting document paragraph {i}" for i in range(10)]
        state = CognitiveState(
            original_goal="Goal item",
            current_task="Task item",
            knowledge_items=tuple(knowledge_list),
        )
        selection = cm.build_context(state, budget=budget)

        # Max items is 3: Goal + Task are protected (2), so at most 1 knowledge item can fit
        assert selection.item_count <= 3
        assert len(selection.omitted_items) >= 7

        # Goal and task MUST be among selected
        assert len(selection.get_items_by_source(ContextSource.CURRENT_GOAL)) == 1
        assert len(selection.get_items_by_source(ContextSource.CURRENT_TASK)) == 1

    def test_token_budget_enforced_and_protected_preserved(self):
        cm = StandardContextManager()
        # Tight token budget
        budget = ContextBudget(max_items=10, max_tokens=40)

        long_goal = "Critical Goal: Save system from shutdown"
        state = CognitiveState(
            original_goal=long_goal,
            knowledge_items=("Non-critical knowledge chunk 1", "Non-critical knowledge chunk 2"),
        )
        selection = cm.build_context(state, budget=budget)

        # Protected goal remains despite token constraints
        assert len(selection.get_items_by_source(ContextSource.CURRENT_GOAL)) == 1
        assert selection.selected_items[0].is_protected is True

    def test_oversized_item_compacted_deterministically(self):
        cm = StandardContextManager()
        budget = ContextBudget(max_items=5, max_tokens=2000, max_content_length_per_item=50)

        huge_knowledge = "A" * 300
        state = CognitiveState(
            original_goal="Brief goal",
            knowledge_items=(huge_knowledge,),
        )
        selection = cm.build_context(state, budget=budget)
        kn_items = selection.get_items_by_source(ContextSource.KNOWLEDGE)
        assert len(kn_items) == 1
        assert len(kn_items[0].content) <= 75
        assert "[truncated]" in kn_items[0].content


# =====================================================================
# G. DEDUPLICATION
# =====================================================================

class TestDeduplication:
    def test_duplicate_content_collapsed_preserving_higher_priority(self):
        item1 = ContextItem(
            item_id="item_low",
            source=ContextSource.RECENT_CONVERSATION,
            content="System is running out of disk space",
            priority=ContextPriority.MEDIUM,
            confidence=0.8,
        )
        item2 = ContextItem(
            item_id="item_high",
            source=ContextSource.EXECUTION_RESULT,
            content="System is running out of disk space",
            priority=ContextPriority.HIGHEST,
            confidence=1.0,
        )
        deduped = deduplicate_items_deterministic([item1, item2])
        assert len(deduped) == 1
        assert deduped[0].priority == ContextPriority.HIGHEST
        assert deduped[0].confidence == 1.0
        assert "deduplicated_sources" in deduped[0].metadata

    def test_independent_web_evidence_preserved_across_different_urls(self):
        ev1 = ContextItem(
            item_id="ev_1",
            source=ContextSource.WEB_EVIDENCE,
            content="Python 3.13 released with free-threaded mode",
            metadata={"url": "https://python.org/release-3-13"},
        )
        ev2 = ContextItem(
            item_id="ev_2",
            source=ContextSource.WEB_EVIDENCE,
            content="Python 3.13 released with free-threaded mode",
            metadata={"url": "https://realpython.com/python313"},
        )
        deduped = deduplicate_items_deterministic([ev1, ev2])
        assert len(deduped) == 2


# =====================================================================
# H. VISUAL CONTEXT
# =====================================================================

class TestVisualContext:
    def test_visual_scene_and_grounded_targets_prioritized_in_visual_focus(self):
        screen = ScreenDimensions(1920, 1080)
        box = BoundingBox(100, 200, 80, 30)
        element = VisualElement(
            element_id="btn_submit",
            element_type=ElementType.BUTTON,
            bounding_box=box,
            text="Submit Form",
            confidence=0.98,
        )
        scene = VisualScene(
            source_observation_id="obs_001",
            screen_dimensions=screen,
            elements=(element,),
            active_window_title="Registration Page - Chrome",
            process_name="chrome.exe",
        )
        target = GroundedTarget(
            target_id="gt_001",
            element=element,
            click_coordinate=(140, 215),
            confidence=0.98,
            source_observation_id="obs_001",
            observation_timestamp=time.time(),
        )

        cm = StandardContextManager()
        state = CognitiveState(
            original_goal="Submit registration form",
            visual_scene=scene,
            grounded_candidates=(target,),
            attention_focus=AttentionFocus.VISUAL_TARGET,
        )
        selection = cm.build_context(state)

        vis_items = selection.get_items_by_source(ContextSource.VISUAL_SCENE)
        assert len(vis_items) >= 2
        # Main scene summary is protected when in visual target focus
        assert any(it.is_protected and "Registration Page" in it.content for it in vis_items)
        assert any("Grounded Target 'Submit Form'" in it.content for it in vis_items)


# =====================================================================
# I. RECOVERY CONTEXT
# =====================================================================

class TestRecoveryContext:
    def test_recovery_focus_prioritizes_failure_and_alternatives(self):
        cm = StandardContextManager()
        rec = RecoveryContext(
            original_goal="Upload daily sales spreadsheet",
            current_plan=MagicMock(),
            attempt=2,
            outcome=ExecutionOutcome.FAILURE,
            failure_classification=FailureClassification.EXECUTION_ERROR,
            failure_reason="Remote server returned HTTP 503 Service Unavailable",
            remaining_retry_budget=1,
            remaining_replan_budget=2,
        )
        verif = VerificationResult(
            verified=False,
            status="failed",
            confidence=1.0,
            reason="Upload endpoint 503 failure",
        )
        state = CognitiveState(
            original_goal="Upload daily sales spreadsheet",
            recovery_state=rec,
            verification_state=verif,
            attention_focus=AttentionFocus.RECOVERY,
        )
        selection = cm.build_context(state)

        rec_items = selection.get_items_by_source(ContextSource.RECOVERY)
        verif_items = selection.get_items_by_source(ContextSource.VERIFICATION)
        assert len(rec_items) == 1
        assert rec_items[0].is_protected is True
        assert "HTTP 503" in rec_items[0].content
        assert len(verif_items) == 1
        assert verif_items[0].is_protected is True


# =====================================================================
# J. PRIVACY & SENSITIVITY
# =====================================================================

class TestPrivacyAndSensitivity:
    def test_sensitive_metadata_preserved_and_redacted_in_dict(self):
        cm = StandardContextManager()
        mem_secret = MemoryEntry(
            key="db_password",
            value="super_secret_pass_123",
            category="credentials",
        )
        state = CognitiveState(
            original_goal="Connect to database",
            memory_items=(mem_secret,),
        )
        selection = cm.build_context(state)
        mem_items = selection.get_items_by_source(ContextSource.MEMORY)
        assert len(mem_items) == 1
        assert mem_items[0].is_sensitive is True
        assert mem_items[0].sensitivity in (SensitivityLevel.SECRET, SensitivityLevel.SENSITIVE)

        # to_dict must redact raw content
        item_dict = mem_items[0].to_dict()
        assert item_dict["content"] == "[REDACTED_SENSITIVE]"
        assert "super_secret_pass_123" not in str(item_dict)

    def test_context_selection_reports_sensitive_content(self):
        cm = StandardContextManager()
        state = CognitiveState(
            original_goal="Manage api keys",
            memory_items=("api_key: sk-1234567890abcdef",),
        )
        selection = cm.build_context(state)
        assert selection.has_sensitive_content() is True


# =====================================================================
# K. MODEL ROUTER INTEGRATION
# =====================================================================

class TestModelRouterIntegration:
    def test_requirement_mapper_elevates_privacy_when_context_is_sensitive(self):
        item = ContextItem(
            item_id="sens_1",
            source=ContextSource.MEMORY,
            content="Sensitive user secret",
            sensitivity=SensitivityLevel.SECRET,
        )
        selection = ContextSelection(
            selected_items=(item,),
            total_candidates=1,
            attention_focus=AttentionFocus.GOAL,
        )
        req = ReasoningRequest(
            goal="Process sensitive data",
            context_selection=selection,
        )

        requirements = derive_requirements_from_request(req)
        assert requirements.locality_requirement == LocalityRequirement.LOCAL_ONLY
        assert requirements.privacy_requirement == PrivacyClass.STRICT_LOCAL

    def test_requirement_mapper_adds_vision_when_visual_items_selected(self):
        item = ContextItem(
            item_id="vis_1",
            source=ContextSource.VISUAL_SCENE,
            content="Visual desktop screenshot active window",
        )
        selection = ContextSelection(
            selected_items=(item,),
            total_candidates=1,
            attention_focus=AttentionFocus.VISUAL_TARGET,
        )
        req = ReasoningRequest(
            goal="Inspect application",
            context_selection=selection,
        )

        requirements = derive_requirements_from_request(req)
        assert ModelCapabilityType.VISION in requirements.required_capabilities
        assert requirements.structured_output_required is True


# =====================================================================
# L. REASONING ENGINE INTEGRATION
# =====================================================================

class TestReasoningEngineIntegration:
    def test_reasoning_engine_injects_context_selection_into_request(self):
        provider = MockReasoningProvider(
            provider_id="mock_reasoner",
            responses=[
                ReasoningResponse(
                    turn_id="turn_1",
                    outcome=ReasoningOutcome.REPORT_COMPLETION,
                    completion_summary="Done with context",
                )
            ],
        )
        cm = StandardContextManager()
        engine = ReasoningEngine(provider=provider, context_manager=cm)

        cs = CognitiveState(
            original_goal="Verify context injection",
            current_task="Run step 1",
        )

        resp, _ = engine.run_turn(
            goal="Verify context injection",
            task_description="Run step 1",
            cognitive_state=cs,
        )

        assert resp.outcome == ReasoningOutcome.REPORT_COMPLETION
        assert len(provider.received_requests) == 1
        req = provider.received_requests[0]
        assert req.context_selection is not None
        assert req.context_selection.item_count >= 2
        assert len(req.context_selection.get_items_by_source(ContextSource.CURRENT_GOAL)) == 1


# =====================================================================
# M. SECURITY & ARCHITECTURAL BOUNDARIES
# =====================================================================

class TestSecurityAndArchitecturalBoundaries:
    def test_context_manager_has_no_tool_execution_capability(self):
        cm = StandardContextManager()
        # ContextManager must not have executor, tool_registry, or os execution attributes
        assert not hasattr(cm, "executor")
        assert not hasattr(cm, "execute")
        assert not hasattr(cm, "tool_registry")
        assert not hasattr(cm, "run_tool")

    def test_context_manager_does_not_mutate_memory_service(self):
        cm = StandardContextManager()
        assert not hasattr(cm, "memory_service")
        assert not hasattr(cm, "add_message")
        assert not hasattr(cm, "save_preference")

    def test_context_manager_does_not_select_model(self):
        cm = StandardContextManager()
        assert not hasattr(cm, "route")
        assert not hasattr(cm, "model_router")
        assert not hasattr(cm, "select_model")

    def test_context_manager_does_not_call_llm(self):
        cm = StandardContextManager()
        assert not hasattr(cm, "ask_ollama")
        assert not hasattr(cm, "generate")
        assert not hasattr(cm, "call_llm")

    def test_context_manager_produces_structured_selection_not_executable_action(self):
        cm = StandardContextManager()
        state = CognitiveState(original_goal="Sample goal")
        selection = cm.build_context(state)
        assert isinstance(selection, ContextSelection)
        assert not hasattr(selection, "execute")
        assert not hasattr(selection, "action_type")


# =====================================================================
# N. DETERMINISM
# =====================================================================

class TestDeterminism:
    def test_identical_cognitive_state_produces_identical_selection(self):
        cm = StandardContextManager()
        state = CognitiveState(
            original_goal="Deterministic goal verification",
            current_task="Verify ordering",
            available_capabilities=("bash", "fs", "computer"),
            knowledge_items=("Chunk A content", "Chunk B content"),
            attention_focus=AttentionFocus.PLANNING,
        )
        selection1 = cm.build_context(state)
        selection2 = cm.build_context(state)

        assert selection1.item_count == selection2.item_count
        for it1, it2 in zip(selection1.selected_items, selection2.selected_items):
            assert it1.source == it2.source
            assert it1.content == it2.content
            assert it1.priority == it2.priority
            assert it1.relevance == it2.relevance
            assert it1.is_protected == it2.is_protected
