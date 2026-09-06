from typing import Optional, List, Dict, Any, Tuple
import copy

from core.interfaces.recovery_interface import RecoveryPlannerInterface
from core.models.plan import Plan
from core.models.task import Task
from core.models.result import Result
from core.models.verification import VerificationResult
from core.models.recovery import (
    RecoveryContext,
    RecoveryDecision,
    RecoveryAction,
    FailureClassification,
    ExecutionOutcome,
)
from brain.recovery.failure_classifier import FailureClassifier


def compute_plan_signature(plan: Plan) -> Tuple[Tuple[str, str, Tuple[Tuple[str, str], ...]], ...]:
    """
    Deterministic signature of a plan's task steps to detect and prevent repeating failed strategies.
    """
    sig = []
    for step in plan.steps:
        tool = getattr(step, "tool", "") or ""
        action = getattr(step, "action", "") or ""
        params = getattr(step, "parameters", {}) or {}
        filtered = {
            k: str(v)
            for k, v in params.items()
            if k not in ("memory_service", "web_provider", "history", "session_id")
        }
        sig.append((tool.lower(), action.lower(), tuple(sorted(filtered.items()))))
    return tuple(sig)


class StandardRecoveryPlanner(RecoveryPlannerInterface):
    """
    Deterministic, model-neutral recovery planner implementing RecoveryPlannerInterface.
    Strictly preserves:
    1. ORIGINAL GOAL != CURRENT PLAN
    2. Zero policy bypass
    3. Anti-loop / repetition protection
    4. Deterministic budget enforcement
    """

    def __init__(self, classifier: Optional[FailureClassifier] = None):
        self.classifier = classifier or FailureClassifier()

    def classify_failure(
        self,
        plan: Plan,
        results: List[Result],
        verification: VerificationResult,
    ) -> FailureClassification:
        return self.classifier.classify(plan, results, verification)

    def decide_recovery(self, context: RecoveryContext) -> RecoveryDecision:
        # 1. Immediate completion if verified success
        if context.outcome == ExecutionOutcome.SUCCESS:
            return RecoveryDecision(
                action=RecoveryAction.CONTINUE,
                classification=FailureClassification.TRANSIENT,  # dummy for success
                reason="Execution completed and verified.",
            )

        # 2. Hard budget & bounds checks
        if context.attempt >= context.limits.max_attempts:
            return RecoveryDecision(
                action=RecoveryAction.ABORT,
                classification=context.failure_classification or FailureClassification.EXECUTION_ERROR,
                reason=f"Maximum execution attempts reached ({context.limits.max_attempts}).",
            )

        if len(context.plan_history) >= context.limits.max_total_recovery_steps:
            return RecoveryDecision(
                action=RecoveryAction.ABORT,
                classification=context.failure_classification or FailureClassification.EXECUTION_ERROR,
                reason=f"Maximum recovery steps reached ({context.limits.max_total_recovery_steps}).",
            )

        classification = context.failure_classification or FailureClassification.EXECUTION_ERROR

        # 3. Permission & Confirmation Required: Cannot proceed autonomously
        if classification in (
            FailureClassification.PERMISSION_REQUIRED,
            FailureClassification.CONFIRMATION_REQUIRED,
        ):
            return RecoveryDecision(
                action=RecoveryAction.ASK_USER,
                classification=classification,
                reason=f"User authorization required: {context.failure_reason}",
                metadata={"requires_user_interaction": True},
            )

        # 4. Policy Denied: Never retry blindly; check for policy-compliant alternative
        if classification == FailureClassification.POLICY_DENIED:
            if context.replan_count < context.limits.max_replans:
                revised_plan = self.create_replan(context)
                if revised_plan is not None:
                    return RecoveryDecision(
                        action=RecoveryAction.REPLAN,
                        classification=classification,
                        reason="Replanning with policy-compliant alternative capability.",
                        revised_plan=revised_plan,
                    )
            return RecoveryDecision(
                action=RecoveryAction.ABORT,
                classification=classification,
                reason=f"Policy denial cannot be bypassed: {context.failure_reason}",
                metadata={"policy_blocked": True},
            )

        # 5. Transient Failures: Retry within bounds
        if classification == FailureClassification.TRANSIENT:
            failed_task_key = self._get_failed_task_key(context.current_plan, context.results)
            current_retries = context.action_retry_counts.get(failed_task_key, 0)

            if current_retries < context.limits.max_retries_per_action:
                return RecoveryDecision(
                    action=RecoveryAction.RETRY,
                    classification=classification,
                    reason=(
                        f"Retrying transient failure on '{failed_task_key}' "
                        f"(attempt {current_retries + 1}/{context.limits.max_retries_per_action})."
                    ),
                    metadata={"retry_target": failed_task_key},
                )
            else:
                # Retries exhausted -> fallback to REPLAN if replan budget remains
                if context.replan_count < context.limits.max_replans:
                    revised_plan = self.create_replan(context)
                    if revised_plan is not None:
                        return RecoveryDecision(
                            action=RecoveryAction.REPLAN,
                            classification=classification,
                            reason="Transient retries exhausted; replanning with alternative strategy.",
                            revised_plan=revised_plan,
                        )
                return RecoveryDecision(
                    action=RecoveryAction.ABORT,
                    classification=classification,
                    reason="Transient retry budget exhausted.",
                )

        # 6. Insufficient Result or Partial Success: Replan targeting unmet requirement
        if classification == FailureClassification.INSUFFICIENT_RESULT or context.outcome == ExecutionOutcome.PARTIAL_SUCCESS:
            if context.replan_count < context.limits.max_replans:
                revised_plan = self.create_replan(context)
                if revised_plan is not None:
                    return RecoveryDecision(
                        action=RecoveryAction.REPLAN,
                        classification=classification,
                        reason="Replanning to satisfy unmet goal requirements.",
                        revised_plan=revised_plan,
                    )
            return RecoveryDecision(
                action=RecoveryAction.ABORT,
                classification=classification,
                reason="Cannot generate revised plan for insufficient results; budget exhausted.",
            )

        # 7. Unsupported or Malformed: Attempt replan or abort
        if classification in (FailureClassification.UNSUPPORTED, FailureClassification.MALFORMED):
            if context.replan_count < context.limits.max_replans:
                revised_plan = self.create_replan(context)
                if revised_plan is not None:
                    return RecoveryDecision(
                        action=RecoveryAction.REPLAN,
                        classification=classification,
                        reason=f"Replanning around {classification.value} issue.",
                        revised_plan=revised_plan,
                    )
            return RecoveryDecision(
                action=RecoveryAction.ABORT,
                classification=classification,
                reason=f"Cannot recover from {classification.value} action: {context.failure_reason}",
            )

        # 8. General Execution Error: Replan if budget remains, else abort
        if context.replan_count < context.limits.max_replans:
            revised_plan = self.create_replan(context)
            if revised_plan is not None:
                return RecoveryDecision(
                    action=RecoveryAction.REPLAN,
                    classification=classification,
                    reason="Replanning with alternative strategy after execution error.",
                    revised_plan=revised_plan,
                )

        return RecoveryDecision(
            action=RecoveryAction.ABORT,
            classification=classification,
            reason=f"Execution error unrecoverable: {context.failure_reason}",
        )

    def create_replan(self, context: RecoveryContext) -> Optional[Plan]:
        """
        Generate an alternative Plan targeting context.original_goal.
        Enforces anti-loop protection by rejecting strategies whose signature matches
        any previous failed attempt in plan_history.
        """
        failed_signatures = {
            compute_plan_signature(entry.plan) for entry in context.plan_history
        }
        current_sig = compute_plan_signature(context.current_plan)
        failed_signatures.add(current_sig)

        goal = context.original_goal
        plan = context.current_plan

        # Identify the failed step
        failed_idx = next(
            (i for i, r in enumerate(context.results) if not getattr(r, "success", True)),
            0,
        )
        failed_task = plan.steps[failed_idx] if failed_idx < len(plan.steps) else None

        tool_name = getattr(failed_task, "tool", "") or ""
        action_name = getattr(failed_task, "action", "") or ""
        params = getattr(failed_task, "parameters", {}) or {}

        candidate_plans: List[Plan] = []

        # Candidate 1: Web search query adaptation or knowledge fallback
        if tool_name == "web":
            query = params.get("query", goal)
            # 1a. Query facet / refinement
            refined_query = f"{query} documentation" if "documentation" not in query else f"{query} overview"
            p1 = Plan(
                goal=goal,
                steps=[
                    Task(
                        id=1,
                        type="web",
                        action="Web Search",
                        tool="web",
                        parameters={"action": "search", "query": refined_query},
                        status="pending",
                    )
                ],
                status="pending",
                confidence=0.85,
            )
            candidate_plans.append(p1)

            # 1b. Knowledge retrieval fallback
            p2 = Plan(
                goal=goal,
                steps=[
                    Task(
                        id=1,
                        type="knowledge",
                        action="Retrieve Knowledge",
                        tool="knowledge",
                        parameters={"query": query},
                        status="pending",
                    )
                ],
                status="pending",
                confidence=0.8,
            )
            candidate_plans.append(p2)

        # Candidate 2: Knowledge retrieval fallback to web search
        elif tool_name == "knowledge":
            query = params.get("query", goal)
            p = Plan(
                goal=goal,
                steps=[
                    Task(
                        id=1,
                        type="web",
                        action="Web Search",
                        tool="web",
                        parameters={"action": "search", "query": query},
                        status="pending",
                    )
                ],
                status="pending",
                confidence=0.85,
            )
            candidate_plans.append(p)

        # Candidate 3: Policy-blocked prohibited tool -> fallback to safe web or knowledge
        elif context.failure_classification == FailureClassification.POLICY_DENIED:
            p = Plan(
                goal=goal,
                steps=[
                    Task(
                        id=1,
                        type="knowledge",
                        action="Retrieve Knowledge",
                        tool="knowledge",
                        parameters={"query": goal},
                        status="pending",
                    )
                ],
                status="pending",
                confidence=0.75,
            )
            candidate_plans.append(p)

        # Candidate 4: Multi-step partial success: keep successful steps, only re-execute failed step
        elif context.outcome == ExecutionOutcome.PARTIAL_SUCCESS or (len(plan.steps) > 1 and failed_idx > 0):
            remaining_steps = []
            for t_idx, step in enumerate(plan.steps[failed_idx:], start=1):
                new_t = Task(
                    id=t_idx,
                    type=step.type,
                    action=step.action,
                    tool=step.tool,
                    parameters=copy.deepcopy(step.parameters),
                    status="pending",
                )
                remaining_steps.append(new_t)
            if remaining_steps:
                p = Plan(
                    goal=goal,
                    steps=remaining_steps,
                    status="pending",
                    confidence=0.9,
                )
                candidate_plans.append(p)

        # Filter candidate plans through anti-loop protection
        for cand in candidate_plans:
            cand_sig = compute_plan_signature(cand)
            if cand_sig not in failed_signatures:
                return cand

        # If all candidates match previously failed signatures, anti-loop triggers: return None
        return None

    def _get_failed_task_key(self, plan: Plan, results: Tuple[Result, ...]) -> str:
        for idx, r in enumerate(results):
            if not getattr(r, "success", True):
                if idx < len(plan.steps):
                    task = plan.steps[idx]
                    return f"{getattr(task, 'tool', 'task')}.{getattr(task, 'action', 'action')}"
                return f"task_{idx}"
        return "unknown_task"
