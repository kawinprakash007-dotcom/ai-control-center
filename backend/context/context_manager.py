import time
import uuid
from typing import List, Optional, Tuple, Dict, Any

from core.interfaces.context_interface import ContextManagerInterface
from core.models.context import (
    CognitiveState,
    ContextItem,
    ContextSource,
    ContextPriority,
    SensitivityLevel,
    AttentionFocus,
    ContextBudget,
    ContextSelection,
)
from core.models.result import Result
from core.models.verification import VerificationResult
from core.models.recovery import RecoveryContext
from core.models.perception import VisualScene, GroundedTarget
from core.models.memory import ChatMessage, MemoryEntry
from core.models.web import EvidenceItem, EvidenceSet, ResearchState
from context.relevance import (
    score_candidate_relevance,
    compute_recency_score,
)
from context.policy import (
    apply_attention_focus_weights,
    deduplicate_items_deterministic,
    compact_and_budget_items,
)


class StandardContextManager(ContextManagerInterface):
    """
    Standard implementation of ContextManagerInterface for AI Control Center.
    Selects, scores, prioritizes, deduplicates, and bounds context for reasoning turns.

    NON-NEGOTIABLE SAFETY GUARANTEES:
    - Purely in-memory and deterministic.
    - NEVER executes tools or actions.
    - NEVER writes to or mutates persistent memory stores.
    - NEVER calls remote models or external embeddings.
    - NEVER selects models (ModelRouter responsibility).
    - Preserves privacy metadata and avoids exposing raw secrets.
    """

    def __init__(self, default_budget: Optional[ContextBudget] = None):
        self.default_budget = default_budget or ContextBudget()

    def build_context(
        self,
        cognitive_state: CognitiveState,
        budget: Optional[ContextBudget] = None,
    ) -> ContextSelection:
        effective_budget = budget or self.default_budget
        current_time = time.time()
        candidates: List[ContextItem] = []

        has_active_failure = bool(
            (cognitive_state.verification_state and not cognitive_state.verification_state.verified) or
            cognitive_state.recovery_state is not None
        )
        is_visual_turn = bool(
            cognitive_state.visual_scene is not None or
            cognitive_state.grounded_candidates or
            cognitive_state.attention_focus == AttentionFocus.VISUAL_TARGET
        )

        # -------------------------------------------------------------
        # 1. CURRENT GOAL (Always protected, HIGHEST priority)
        # -------------------------------------------------------------
        candidates.append(
            ContextItem(
                item_id=f"goal_{uuid.uuid4().hex[:8]}",
                source=ContextSource.CURRENT_GOAL,
                content=cognitive_state.original_goal,
                relevance=1.0,
                priority=ContextPriority.HIGHEST,
                recency=1.0,
                confidence=1.0,
                timestamp=current_time,
                is_protected=True,
                metadata={"source_type": "original_goal"},
            )
        )

        # -------------------------------------------------------------
        # 2. CURRENT TASK & PLAN STEP (Always protected if present)
        # -------------------------------------------------------------
        if cognitive_state.current_task:
            candidates.append(
                ContextItem(
                    item_id=f"task_{uuid.uuid4().hex[:8]}",
                    source=ContextSource.CURRENT_TASK,
                    content=cognitive_state.current_task,
                    relevance=1.0,
                    priority=ContextPriority.HIGHEST,
                    recency=1.0,
                    confidence=1.0,
                    timestamp=current_time,
                    is_protected=True,
                    metadata={"source_type": "current_task"},
                )
            )

        if cognitive_state.current_plan_step and cognitive_state.current_plan_step != cognitive_state.current_task:
            candidates.append(
                ContextItem(
                    item_id=f"plan_step_{uuid.uuid4().hex[:8]}",
                    source=ContextSource.CURRENT_TASK,
                    content=f"Current step: {cognitive_state.current_plan_step}",
                    relevance=0.9,
                    priority=ContextPriority.HIGH,
                    recency=1.0,
                    confidence=1.0,
                    timestamp=current_time,
                    is_protected=False,
                    metadata={"source_type": "plan_step"},
                )
            )

        # -------------------------------------------------------------
        # 3. VERIFICATION STATE
        # -------------------------------------------------------------
        if cognitive_state.verification_state is not None:
            v = cognitive_state.verification_state
            v_content = f"Verification {v.status.upper()} (verified={v.verified}, confidence={v.confidence:.2f}): {v.reason}"
            if v.failed_task_id is not None:
                v_content += f" [Failed Task ID: {v.failed_task_id}]"

            # Failed verification is an active blocker -> PROTECTED
            is_v_protected = not v.verified
            candidates.append(
                ContextItem(
                    item_id=f"verif_{uuid.uuid4().hex[:8]}",
                    source=ContextSource.VERIFICATION,
                    content=v_content,
                    relevance=1.0 if not v.verified else 0.85,
                    priority=ContextPriority.HIGHEST if not v.verified else ContextPriority.HIGH,
                    recency=1.0,
                    confidence=v.confidence,
                    timestamp=current_time,
                    is_protected=is_v_protected,
                    metadata={"verified": v.verified, "status": v.status},
                )
            )

        # -------------------------------------------------------------
        # 4. RECOVERY CONTEXT
        # -------------------------------------------------------------
        if cognitive_state.recovery_state is not None:
            rec = cognitive_state.recovery_state
            outcome_val = rec.outcome.value if hasattr(rec.outcome, "value") else str(rec.outcome)
            fail_cls = rec.failure_classification.value if rec.failure_classification and hasattr(rec.failure_classification, "value") else str(rec.failure_classification or "none")
            rec_content = (
                f"RECOVERY ATTEMPT {rec.attempt} [Outcome: {outcome_val}, Classification: {fail_cls}]: "
                f"Failure reason: '{rec.failure_reason or 'unspecified'}'. "
                f"Budgets remaining - Retries: {rec.remaining_retry_budget}, Replans: {rec.remaining_replan_budget}."
            )
            # Active failure recovery is critical -> PROTECTED
            candidates.append(
                ContextItem(
                    item_id=f"rec_{uuid.uuid4().hex[:8]}",
                    source=ContextSource.RECOVERY,
                    content=rec_content,
                    relevance=1.0,
                    priority=ContextPriority.HIGHEST,
                    recency=1.0,
                    confidence=1.0,
                    timestamp=current_time,
                    is_protected=True,
                    metadata={"attempt": rec.attempt, "outcome": outcome_val},
                )
            )

        # -------------------------------------------------------------
        # 5. RECENT EXECUTION OUTCOMES
        # -------------------------------------------------------------
        for idx, outcome in enumerate(cognitive_state.recent_execution_outcomes):
            if isinstance(outcome, Result):
                cap = outcome.capability or "operation"
                act = outcome.action or "action"
                status_str = "SUCCESS" if outcome.success else "FAILURE"
                out_text = f"Action {cap}.{act}: {status_str} - {outcome.message}"
                if outcome.output:
                    out_text += f" Output: {outcome.output[:200]}"
                out_rel = 1.0 if not outcome.success else 0.8
                out_pri = ContextPriority.HIGHEST if not outcome.success else ContextPriority.HIGH
                candidates.append(
                    ContextItem(
                        item_id=f"exec_{idx}_{uuid.uuid4().hex[:6]}",
                        source=ContextSource.EXECUTION_RESULT,
                        content=out_text,
                        relevance=out_rel,
                        priority=out_pri,
                        recency=0.9,
                        confidence=1.0,
                        timestamp=current_time,
                        is_protected=(not outcome.success),
                        metadata={"capability": cap, "action": act, "success": outcome.success},
                    )
                )
            elif isinstance(outcome, dict):
                candidates.append(
                    ContextItem(
                        item_id=f"exec_dict_{idx}_{uuid.uuid4().hex[:6]}",
                        source=ContextSource.EXECUTION_RESULT,
                        content=str(outcome),
                        relevance=0.7,
                        priority=ContextPriority.HIGH,
                        recency=0.8,
                        confidence=0.9,
                        timestamp=current_time,
                        is_protected=False,
                        metadata=dict(outcome),
                    )
                )

        # -------------------------------------------------------------
        # 6. VISUAL SCENE & GROUNDED TARGETS
        # -------------------------------------------------------------
        if cognitive_state.visual_scene is not None:
            scene = cognitive_state.visual_scene
            win_title = scene.active_window_title or "unknown window"
            proc_name = scene.process_name or "unknown process"
            scene_summary = (
                f"Visual Scene: Active window '{win_title}' ({proc_name}), "
                f"screen {scene.screen_dimensions.width}x{scene.screen_dimensions.height}, "
                f"{scene.element_count} UI elements detected."
            )
            candidates.append(
                ContextItem(
                    item_id=f"scene_{uuid.uuid4().hex[:8]}",
                    source=ContextSource.VISUAL_SCENE,
                    content=scene_summary,
                    relevance=0.95 if is_visual_turn else 0.6,
                    priority=ContextPriority.HIGHEST if cognitive_state.attention_focus == AttentionFocus.VISUAL_TARGET else ContextPriority.HIGH,
                    recency=1.0,
                    confidence=1.0,
                    timestamp=scene.timestamp or current_time,
                    is_protected=(cognitive_state.attention_focus == AttentionFocus.VISUAL_TARGET),
                    metadata={"window": win_title, "process": proc_name, "observation_id": scene.source_observation_id},
                )
            )

            # Extract bounded interactive elements matching goal/task
            matching_elements = []
            for q in [cognitive_state.original_goal, cognitive_state.current_task or ""]:
                if q:
                    matching_elements.extend(scene.find_text(q, exact=False))
            # Dedup elements
            seen_el_ids = set()
            for el in matching_elements[:5]:
                if el.element_id not in seen_el_ids:
                    seen_el_ids.add(el.element_id)
                    el_desc = f"UI Element [{el.element_type.value.upper()}] '{el.text}' at ({el.center[0]}, {el.center[1]}), confidence={el.confidence:.2f}"
                    candidates.append(
                        ContextItem(
                            item_id=f"el_{el.element_id}_{uuid.uuid4().hex[:4]}",
                            source=ContextSource.VISUAL_SCENE,
                            content=el_desc,
                            relevance=0.85,
                            priority=ContextPriority.HIGH,
                            recency=0.9,
                            confidence=el.confidence,
                            timestamp=scene.timestamp or current_time,
                            metadata={"element_id": el.element_id, "type": el.element_type.value},
                        )
                    )

        for g_idx, target in enumerate(cognitive_state.grounded_candidates[:5]):
            target_desc = (
                f"Grounded Target '{target.element.text or target.target_id}' "
                f"at click coordinates ({target.click_coordinate[0]}, {target.click_coordinate[1]}), "
                f"confidence={target.confidence:.2f}"
            )
            candidates.append(
                ContextItem(
                    item_id=f"target_{g_idx}_{uuid.uuid4().hex[:6]}",
                    source=ContextSource.VISUAL_SCENE,
                    content=target_desc,
                    relevance=0.9,
                    priority=ContextPriority.HIGH,
                    recency=0.95,
                    confidence=target.confidence,
                    timestamp=target.observation_timestamp or current_time,
                    metadata={"target_id": target.target_id, "coords": target.click_coordinate},
                )
            )

        # -------------------------------------------------------------
        # 7. WEB EVIDENCE & RESEARCH STATE
        # -------------------------------------------------------------
        evidence_items = []
        if isinstance(cognitive_state.web_evidence, EvidenceSet):
            evidence_items = list(cognitive_state.web_evidence.items)
        elif isinstance(cognitive_state.web_evidence, (list, tuple)):
            evidence_items = list(cognitive_state.web_evidence)

        for ev_idx, ev in enumerate(evidence_items):
            if isinstance(ev, EvidenceItem):
                ev_content = f"[{ev.domain}] {ev.title}: {ev.content}"
                rel = score_candidate_relevance(
                    content=ev_content,
                    goal=cognitive_state.original_goal,
                    task=cognitive_state.current_task,
                    source=ContextSource.WEB_EVIDENCE,
                    confidence=0.85,
                )
                candidates.append(
                    ContextItem(
                        item_id=f"ev_{ev.id}",
                        source=ContextSource.WEB_EVIDENCE,
                        content=ev_content,
                        relevance=rel,
                        priority=ContextPriority.HIGH if cognitive_state.attention_focus == AttentionFocus.RESEARCH else ContextPriority.MEDIUM,
                        recency=0.8,
                        confidence=0.85,
                        timestamp=current_time,
                        metadata={"url": ev.url, "domain": ev.domain, "id": ev.id},
                    )
                )
            elif isinstance(ev, dict):
                candidates.append(
                    ContextItem(
                        item_id=f"ev_dict_{ev_idx}_{uuid.uuid4().hex[:6]}",
                        source=ContextSource.WEB_EVIDENCE,
                        content=str(ev.get("content", ev)),
                        relevance=0.6,
                        priority=ContextPriority.MEDIUM,
                        recency=0.7,
                        confidence=0.8,
                        timestamp=current_time,
                        metadata=dict(ev),
                    )
                )

        if cognitive_state.research_state is not None:
            r = cognitive_state.research_state
            if isinstance(r, ResearchState):
                r_content = (
                    f"Research State on '{r.objective}': status={r.status}, iteration={r.iteration}, "
                    f"evidence count={r.evidence_count}."
                )
                candidates.append(
                    ContextItem(
                        item_id=f"res_{uuid.uuid4().hex[:8]}",
                        source=ContextSource.RESEARCH,
                        content=r_content,
                        relevance=0.8,
                        priority=ContextPriority.HIGH if cognitive_state.attention_focus == AttentionFocus.RESEARCH else ContextPriority.MEDIUM,
                        recency=0.9,
                        confidence=0.9,
                        timestamp=current_time,
                        metadata={"status": r.status, "objective": r.objective},
                    )
                )

        # -------------------------------------------------------------
        # 8. MEMORY & CONVERSATION ITEMS (With sensitivity tagging)
        # -------------------------------------------------------------
        for m_idx, mem in enumerate(cognitive_state.memory_items):
            if isinstance(mem, MemoryEntry):
                mem_text = f"User preference/fact [{mem.category}] {mem.key}: {mem.value}"
                # Check for sensitive keywords
                is_sens = any(k in mem.key.lower() or k in str(mem.value).lower() for k in ["password", "secret", "token", "credential", "private", "api_key"])
                sens_level = SensitivityLevel.SECRET if is_sens else SensitivityLevel.PUBLIC
                rel = score_candidate_relevance(
                    content=mem_text,
                    goal=cognitive_state.original_goal,
                    task=cognitive_state.current_task,
                    source=ContextSource.MEMORY,
                )
                candidates.append(
                    ContextItem(
                        item_id=f"mem_pref_{mem.key}_{uuid.uuid4().hex[:4]}",
                        source=ContextSource.MEMORY,
                        content=mem_text,
                        relevance=rel,
                        priority=ContextPriority.HIGH if cognitive_state.attention_focus == AttentionFocus.MEMORY_RECALL else ContextPriority.MEDIUM,
                        recency=0.6,
                        confidence=0.95,
                        sensitivity=sens_level,
                        timestamp=mem.updated_at.timestamp() if hasattr(mem.updated_at, "timestamp") else current_time,
                        metadata={"key": mem.key, "category": mem.category},
                    )
                )
            elif isinstance(mem, ChatMessage):
                c_text = f"{mem.role.value.capitalize()}: {mem.content}"
                is_sens = any(k in mem.content.lower() for k in ["password", "secret", "token", "credential", "private", "api_key"])
                sens_level = SensitivityLevel.SECRET if is_sens else SensitivityLevel.PUBLIC
                ts = mem.timestamp.timestamp() if hasattr(mem.timestamp, "timestamp") else current_time
                rec_score = compute_recency_score(ts, current_time)
                rel = score_candidate_relevance(
                    content=mem.content,
                    goal=cognitive_state.original_goal,
                    task=cognitive_state.current_task,
                    source=ContextSource.RECENT_CONVERSATION,
                    recency_score=rec_score,
                )
                candidates.append(
                    ContextItem(
                        item_id=f"chat_{mem.id}",
                        source=ContextSource.RECENT_CONVERSATION,
                        content=c_text,
                        relevance=rel,
                        priority=ContextPriority.HIGH if cognitive_state.attention_focus == AttentionFocus.MEMORY_RECALL else ContextPriority.MEDIUM,
                        recency=rec_score,
                        confidence=0.9,
                        sensitivity=sens_level,
                        timestamp=ts,
                        metadata={"session_id": mem.session_id, "role": mem.role.value},
                    )
                )
            elif isinstance(mem, str):
                is_sens = any(k in mem.lower() for k in ["password", "secret", "token", "credential", "private", "api_key"])
                sens_level = SensitivityLevel.SECRET if is_sens else SensitivityLevel.PUBLIC
                rel = score_candidate_relevance(
                    content=mem,
                    goal=cognitive_state.original_goal,
                    task=cognitive_state.current_task,
                    source=ContextSource.MEMORY,
                )
                candidates.append(
                    ContextItem(
                        item_id=f"mem_str_{m_idx}_{uuid.uuid4().hex[:6]}",
                        source=ContextSource.MEMORY,
                        content=mem,
                        relevance=rel,
                        priority=ContextPriority.MEDIUM,
                        recency=0.5,
                        confidence=0.85,
                        sensitivity=sens_level,
                        timestamp=current_time,
                    )
                )

        # -------------------------------------------------------------
        # 9. KNOWLEDGE ITEMS
        # -------------------------------------------------------------
        for k_idx, kn in enumerate(cognitive_state.knowledge_items):
            kn_content = kn.content if hasattr(kn, "content") else str(kn)
            rel = score_candidate_relevance(
                content=kn_content,
                goal=cognitive_state.original_goal,
                task=cognitive_state.current_task,
                source=ContextSource.KNOWLEDGE,
            )
            candidates.append(
                ContextItem(
                    item_id=f"kn_{k_idx}_{uuid.uuid4().hex[:6]}",
                    source=ContextSource.KNOWLEDGE,
                    content=kn_content,
                    relevance=rel,
                    priority=ContextPriority.MEDIUM,
                    recency=0.4,
                    confidence=0.85,
                    timestamp=current_time,
                    metadata={"chunk_id": getattr(kn, "chunk_id", None)},
                )
            )

        # -------------------------------------------------------------
        # 10. CAPABILITY STATE
        # -------------------------------------------------------------
        if cognitive_state.available_capabilities:
            cap_list = ", ".join(sorted(cognitive_state.available_capabilities))
            candidates.append(
                ContextItem(
                    item_id=f"caps_{uuid.uuid4().hex[:8]}",
                    source=ContextSource.CAPABILITY_STATE,
                    content=f"Available system capabilities: [{cap_list}]",
                    relevance=0.7,
                    priority=ContextPriority.HIGH if cognitive_state.attention_focus in (AttentionFocus.PLANNING, AttentionFocus.EXECUTION) else ContextPriority.MEDIUM,
                    recency=1.0,
                    confidence=1.0,
                    timestamp=current_time,
                    metadata={"capabilities": list(cognitive_state.available_capabilities)},
                )
            )

        # -------------------------------------------------------------
        # 11. DEDUPLICATION
        # -------------------------------------------------------------
        total_candidate_count = len(candidates)
        deduped = deduplicate_items_deterministic(candidates)

        # -------------------------------------------------------------
        # 12. DETERMINISTIC RANKING & ATTENTION FOCUS WEIGHTING
        # -------------------------------------------------------------
        # Sort key: (composite_score, priority, relevance, recency, content_length_negative, item_id)
        # Content length negative breaks ties in favor of more compact items.
        sorted_candidates = sorted(
            deduped,
            key=lambda it: (
                apply_attention_focus_weights(it, cognitive_state.attention_focus),
                int(it.priority),
                it.relevance,
                it.recency,
                -len(it.content),
                it.item_id,
            ),
            reverse=True,
        )

        # -------------------------------------------------------------
        # 13. BUDGET ENFORCEMENT & COMPACTION
        # -------------------------------------------------------------
        selected, omitted = compact_and_budget_items(sorted_candidates, effective_budget)

        selection_reason = (
            f"Selected {len(selected)} items (~{sum(it.token_estimate for it in selected)} tokens) "
            f"from {total_candidate_count} candidates under attention focus '{cognitive_state.attention_focus.value}'."
        )

        return ContextSelection(
            selected_items=tuple(selected),
            omitted_items=tuple(omitted),
            total_candidates=total_candidate_count,
            selection_reason=selection_reason,
            budget=effective_budget,
            attention_focus=cognitive_state.attention_focus,
            provenance_metadata={
                "goal": cognitive_state.original_goal,
                "task": cognitive_state.current_task,
                "attention_focus": cognitive_state.attention_focus.value,
                "protected_count": sum(1 for it in selected if it.is_protected),
                "has_sensitive": any(it.is_sensitive for it in selected),
            },
        )
