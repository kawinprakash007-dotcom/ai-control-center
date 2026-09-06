import time
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.interfaces.anticipation_interface import (
    AnticipatoryAnalyzerInterface,
    EvidenceEvaluatorInterface,
)
from core.models.anticipation import (
    Anticipation,
    AnticipationProvenance,
    AnticipationStatus,
    EvidenceItem,
    EvidenceSourceType,
    FutureConditionType,
    TimeHorizon,
)
from core.models.autonomy import Event, EventCategory
from core.models.goal import Goal
from core.models.world_state import WorldState
from anticipation.evaluator import DeterministicEvidenceEvaluator


class DeterministicAnticipatoryAnalyzer(AnticipatoryAnalyzerInterface):
    """
    Model-neutral, deterministic anticipatory analyzer for Phase 4.6.
    Inspects current world state, active goals, recent events, and evidence
    to formulate explainable, bounded candidate anticipation hypotheses.
    
    CRITICAL ARCHITECTURAL INVARIANT:
    Hypotheses are NEVER written into World State as observed facts.
    Never executes tools, shell commands, or models directly.
    """

    def __init__(
        self,
        evaluator: Optional[EvidenceEvaluatorInterface] = None,
        clock: Optional[Any] = None,
        max_candidates: int = 50,
    ):
        self.evaluator = evaluator or DeterministicEvidenceEvaluator(clock=clock)
        self.clock = clock or time.time
        self.max_candidates = max_candidates

    def analyze(
        self,
        world_state: Optional[Any],
        active_goals: Sequence[Any],
        recent_events: Sequence[Any],
        evidence: Sequence[EvidenceItem],
        horizon_limit: Optional[TimeHorizon] = None,
        now: Optional[float] = None,
    ) -> List[Anticipation]:
        current_time = now if now is not None else self.clock()
        candidates: List[Anticipation] = []

        # -------------------------------------------------------------
        # 1. Analyze Current World State for Resource / Environment Risks
        # -------------------------------------------------------------
        if world_state is not None:
            candidates.extend(
                self._analyze_world_state(world_state, active_goals, current_time)
            )

        # -------------------------------------------------------------
        # 2. Analyze Active Goals for Deadline / Completion Risks
        # -------------------------------------------------------------
        if active_goals:
            candidates.extend(
                self._analyze_active_goals(active_goals, current_time)
            )

        # -------------------------------------------------------------
        # 3. Analyze Recent Events for Recurring Safety / Failure Risks
        # -------------------------------------------------------------
        if recent_events:
            candidates.extend(
                self._analyze_recent_events(recent_events, active_goals, current_time)
            )

        # -------------------------------------------------------------
        # 4. Integrate Explicit External Evidence Items
        # -------------------------------------------------------------
        if evidence:
            candidates.extend(
                self._analyze_explicit_evidence(evidence, active_goals, current_time)
            )

        # Bounded candidate limit
        trimmed = candidates[:self.max_candidates]

        # Apply horizon filter if requested
        if horizon_limit:
            filtered = [c for c in trimmed if c.horizon == horizon_limit]
            return filtered

        return trimmed

    def _analyze_world_state(
        self,
        world_state: Any,
        active_goals: Sequence[Any],
        now: float,
    ) -> List[Anticipation]:
        """Extract resource depletion or environment risk hypotheses from world state."""
        results: List[Anticipation] = []
        conditions = getattr(world_state, "conditions", ())
        if not conditions:
            return results

        for cond in conditions:
            prop_name = getattr(cond, "property_name", "")
            prop_lower = prop_name.lower()
            entity_id = getattr(cond, "entity_id", "")
            val = getattr(cond, "value", None)

            # Detect Resource Depletion Risks
            if any(k in prop_lower for k in ("battery", "fuel", "disk", "memory", "power", "water", "charge")):
                if isinstance(val, (int, float)) and 0 < val <= 35:
                    horizon = TimeHorizon.NEAR_TERM
                    window = horizon.get_default_window_seconds()
                    ev_item = EvidenceItem(
                        evidence_id=f"ev_ws_{uuid.uuid4().hex[:8]}",
                        source_type=EvidenceSourceType.WORLD_STATE,
                        source_id=f"{entity_id}.{prop_name}",
                        description=f"Entity '{entity_id}' property '{prop_name}' is low at {val}%.",
                        confidence=getattr(cond, "confidence", 0.9),
                        observed_at=getattr(cond, "observed_at", now),
                        metadata={"entity_id": entity_id, "property": prop_name, "value": val},
                    )
                    conf, fresh, rel = self.evaluator.evaluate_evidence([ev_item], active_goals, now=now)
                    results.append(
                        Anticipation(
                            anticipation_id=f"ant_deplete_{uuid.uuid4().hex[:10]}",
                            condition_type=FutureConditionType.RESOURCE_DEPLETION_RISK,
                            description=f"Resource '{prop_name}' on '{entity_id}' is depleting ({val}%); projected insufficient within {int(window[1]/60)}m.",
                            target_entity_id=entity_id,
                            hypothetical_state={prop_name: max(0.0, float(val) - 20.0)},
                            horizon=horizon,
                            horizon_window_seconds=window,
                            confidence=conf,
                            relevance=rel,
                            freshness=fresh,
                            evidence_items=(ev_item,),
                            provenance=AnticipationProvenance(
                                source_entity="DeterministicAnticipatoryAnalyzer",
                                created_at=now,
                                correlation_id=f"corr_ant_{uuid.uuid4().hex[:8]}",
                            ),
                            correlation_id=f"corr_ant_{uuid.uuid4().hex[:8]}",
                            metadata={"risk_threshold": 25.0, "current_value": val},
                        )
                    )

            # Detect Environmental Risks
            elif any(k in prop_lower for k in ("temp", "temperature", "heat", "pressure", "vibration")):
                if isinstance(val, (int, float)) and val >= 75:
                    horizon = TimeHorizon.IMMEDIATE
                    window = horizon.get_default_window_seconds()
                    ev_item = EvidenceItem(
                        evidence_id=f"ev_env_{uuid.uuid4().hex[:8]}",
                        source_type=EvidenceSourceType.WORLD_STATE,
                        source_id=f"{entity_id}.{prop_name}",
                        description=f"Environmental metric '{prop_name}' on '{entity_id}' is elevated at {val}.",
                        confidence=getattr(cond, "confidence", 0.85),
                        observed_at=getattr(cond, "observed_at", now),
                        metadata={"entity_id": entity_id, "property": prop_name, "value": val},
                    )
                    conf, fresh, rel = self.evaluator.evaluate_evidence([ev_item], active_goals, now=now)
                    results.append(
                        Anticipation(
                            anticipation_id=f"ant_env_{uuid.uuid4().hex[:10]}",
                            condition_type=FutureConditionType.ENVIRONMENT_CHANGE_RISK,
                            description=f"Environmental parameter '{prop_name}' on '{entity_id}' indicates thermal/pressure stress.",
                            target_entity_id=entity_id,
                            hypothetical_state={prop_name: "overheat_hazard"},
                            horizon=horizon,
                            horizon_window_seconds=window,
                            confidence=conf,
                            relevance=rel,
                            freshness=fresh,
                            evidence_items=(ev_item,),
                            provenance=AnticipationProvenance(
                                source_entity="DeterministicAnticipatoryAnalyzer",
                                created_at=now,
                                correlation_id=f"corr_env_{uuid.uuid4().hex[:8]}",
                            ),
                            correlation_id=f"corr_env_{uuid.uuid4().hex[:8]}",
                            metadata={"current_value": val},
                        )
                    )
        return results

    def _analyze_active_goals(
        self,
        active_goals: Sequence[Any],
        now: float,
    ) -> List[Anticipation]:
        """Extract deadline or goal completion risks from active goal progress."""
        results: List[Anticipation] = []
        for g in active_goals:
            gid = getattr(g, "goal_id", "")
            deadline = getattr(g, "deadline", None)
            if deadline is not None and deadline > now:
                time_remaining = deadline - now
                if time_remaining <= 1800.0:  # <= 30 mins
                    horizon = TimeHorizon.NEAR_TERM
                    window = (0.0, time_remaining)
                    ev_item = EvidenceItem(
                        evidence_id=f"ev_goal_{uuid.uuid4().hex[:8]}",
                        source_type=EvidenceSourceType.ACTIVE_GOAL,
                        source_id=gid,
                        description=f"Goal '{gid}' deadline expires in {int(time_remaining)}s.",
                        confidence=0.95,
                        observed_at=now,
                        metadata={"goal_id": gid, "deadline": deadline},
                    )
                    conf, fresh, rel = self.evaluator.evaluate_evidence([ev_item], active_goals, now=now)
                    results.append(
                        Anticipation(
                            anticipation_id=f"ant_dl_{uuid.uuid4().hex[:10]}",
                            condition_type=FutureConditionType.DEADLINE_RISK,
                            description=f"Goal '{gid}' has an imminent deadline approaching in {int(time_remaining/60)}m.",
                            target_entity_id=gid,
                            hypothetical_state={"deadline_breach": True, "goal_id": gid},
                            horizon=horizon,
                            horizon_window_seconds=window,
                            confidence=conf,
                            relevance=1.0,  # Directly tied to an active goal
                            freshness=fresh,
                            evidence_items=(ev_item,),
                            related_goal_ids=(gid,),
                            provenance=AnticipationProvenance(
                                source_entity="DeterministicAnticipatoryAnalyzer",
                                created_at=now,
                                correlation_id=f"corr_dl_{uuid.uuid4().hex[:8]}",
                            ),
                            correlation_id=f"corr_dl_{uuid.uuid4().hex[:8]}",
                            metadata={"goal_id": gid, "time_remaining": time_remaining},
                        )
                    )
        return results

    def _analyze_recent_events(
        self,
        recent_events: Sequence[Any],
        active_goals: Sequence[Any],
        now: float,
    ) -> List[Anticipation]:
        """Extract safety or capability risks from recent event trends."""
        results: List[Anticipation] = []
        failures = [e for e in recent_events if getattr(e, "category", None) == EventCategory.FAILURE]
        if len(failures) >= 2:
            horizon = TimeHorizon.IMMEDIATE
            window = horizon.get_default_window_seconds()
            ev_items = tuple(
                EvidenceItem(
                    evidence_id=f"ev_fail_{i}",
                    source_type=EvidenceSourceType.EVENT,
                    source_id=getattr(f, "event_id", str(i)),
                    description=f"Observed recent failure event: {getattr(f, 'event_type', 'error')}",
                    confidence=0.9,
                    observed_at=getattr(f, "timestamp", now),
                )
                for i, f in enumerate(failures[:3])
            )
            conf, fresh, rel = self.evaluator.evaluate_evidence(ev_items, active_goals, now=now)
            results.append(
                Anticipation(
                    anticipation_id=f"ant_safety_{uuid.uuid4().hex[:10]}",
                    condition_type=FutureConditionType.SAFETY_RISK,
                    description=f"Cluster of {len(failures)} recent failure events indicates potential cascading safety risk.",
                    hypothetical_state={"cascading_failure": True},
                    horizon=horizon,
                    horizon_window_seconds=window,
                    confidence=conf,
                    relevance=rel,
                    freshness=fresh,
                    evidence_items=ev_items,
                    related_event_ids=tuple(getattr(f, "event_id", "") for f in failures[:3]),
                    provenance=AnticipationProvenance(
                        source_entity="DeterministicAnticipatoryAnalyzer",
                        created_at=now,
                        correlation_id=f"corr_fail_{uuid.uuid4().hex[:8]}",
                    ),
                    correlation_id=f"corr_fail_{uuid.uuid4().hex[:8]}",
                    metadata={"failure_count": len(failures)},
                )
            )
        return results

    def _analyze_explicit_evidence(
        self,
        evidence: Sequence[EvidenceItem],
        active_goals: Sequence[Any],
        now: float,
    ) -> List[Anticipation]:
        """Formulate anticipation hypothesis from explicit caller-provided evidence."""
        results: List[Anticipation] = []
        if not evidence:
            return results

        conf, fresh, rel = self.evaluator.evaluate_evidence(evidence, active_goals, now=now)
        first_ev = evidence[0]
        horizon = TimeHorizon.NEAR_TERM
        window = horizon.get_default_window_seconds()

        results.append(
            Anticipation(
                anticipation_id=f"ant_ev_{uuid.uuid4().hex[:10]}",
                condition_type=FutureConditionType.CAPABILITY_RISK,
                description=f"Anticipated capability constraint based on evidence: {first_ev.description}",
                target_entity_id=first_ev.source_id,
                hypothetical_state={"constraint": first_ev.description},
                horizon=horizon,
                horizon_window_seconds=window,
                confidence=conf,
                relevance=rel,
                freshness=fresh,
                evidence_items=tuple(evidence),
                provenance=AnticipationProvenance(
                    source_entity="DeterministicAnticipatoryAnalyzer",
                    created_at=now,
                    correlation_id=f"corr_ev_{uuid.uuid4().hex[:8]}",
                ),
                correlation_id=f"corr_ev_{uuid.uuid4().hex[:8]}",
                metadata={"source_id": first_ev.source_id},
            )
        )
        return results
