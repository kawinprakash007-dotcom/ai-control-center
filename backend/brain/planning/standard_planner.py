from typing import Dict, Any

from core.interfaces.decision_planner_interface import DecisionPlannerInterface
from core.models.decision import Decision, CapabilityType, ExecutionMode
from core.models.plan import Plan
from core.models.task import Task


class StandardPlanner(DecisionPlannerInterface):
    """
    Deterministic implementation of DecisionPlannerInterface.
    Translates an immutable Decision into an actionable Plan containing Task steps.
    """

    def plan(self, decision: Decision) -> Plan:
        if not isinstance(decision, Decision):
            raise TypeError(
                f"StandardPlanner.plan expects a Decision instance, got {type(decision).__name__}"
            )

        plan = Plan(
            goal=decision.primary_goal,
            steps=[],
            status="pending",
            confidence=decision.confidence,
        )

        # 1. DIRECT Execution Mode
        if decision.execution_mode == ExecutionMode.DIRECT:
            if decision.primary_goal in ("prompt_user_input", "clarify_request"):
                return plan

            action = "Respond to User"
            params = self._extract_task_parameters(decision)
            task = Task(
                id=1,
                type="chat",
                action=action,
                tool="chat",
                parameters=params,
                status="pending",
            )
            plan.steps.append(task)
            return plan

        # 2. SINGLE_STEP Execution Mode
        if decision.execution_mode == ExecutionMode.SINGLE_STEP:
            if decision.primary_goal == "retrieve_knowledge":
                params = self._extract_task_parameters(decision)
                task = Task(
                    id=1,
                    type="knowledge",
                    action="Retrieve Knowledge",
                    tool="knowledge",
                    parameters=params,
                    status="pending",
                )
                plan.steps.append(task)
                return plan

            if decision.primary_goal == "execute_tool":
                tool_hint = decision.routing_hints.get("tool_hint")
                tool_name = tool_hint if tool_hint else "tool"
                action = "Execute Tool"
                params = self._extract_task_parameters(decision)
                task = Task(
                    id=1,
                    type="tool",
                    action=action,
                    tool=tool_name,
                    parameters=params,
                    status="pending",
                )
                plan.steps.append(task)
                return plan

            if decision.primary_goal == "manage_memory":
                params = self._extract_task_parameters(decision)
                task = Task(
                    id=1,
                    type="memory",
                    action="Manage Memory",
                    tool="memory",
                    parameters=params,
                    status="pending",
                )
                plan.steps.append(task)
                return plan

            # Fallback for single-step by capability
            if decision.required_capabilities:
                task = self._build_task_for_capability(
                    decision.required_capabilities[0], 1, decision
                )
                plan.steps.append(task)
                return plan

            params = self._extract_task_parameters(decision)
            task = Task(
                id=1,
                type="chat",
                action="Respond to User",
                tool="chat",
                parameters=params,
                status="pending",
            )
            plan.steps.append(task)
            return plan

        # 3. MULTI_STEP Execution Mode
        if decision.execution_mode == ExecutionMode.MULTI_STEP:
            for idx, capability in enumerate(decision.required_capabilities, start=1):
                task = self._build_task_for_capability(capability, idx, decision)
                plan.steps.append(task)
            return plan

        # Unsupported execution mode fallback
        raise ValueError(f"Unsupported execution mode: {decision.execution_mode}")

    def _build_task_for_capability(
        self, capability: CapabilityType, task_id: int, decision: Decision
    ) -> Task:
        params = self._extract_task_parameters(decision)

        if capability == CapabilityType.CHAT:
            return Task(
                id=task_id,
                type="chat",
                action="Respond to User",
                tool="chat",
                parameters=params,
                status="pending",
            )

        if capability == CapabilityType.KNOWLEDGE:
            return Task(
                id=task_id,
                type="knowledge",
                action="Retrieve Knowledge",
                tool="knowledge",
                parameters=params,
                status="pending",
            )

        if capability == CapabilityType.TOOL:
            tool_hint = decision.routing_hints.get("tool_hint")
            tool_name = tool_hint if tool_hint else "tool"
            action = "Execute Tool"
            return Task(
                id=task_id,
                type="tool",
                action=action,
                tool=tool_name,
                parameters=params,
                status="pending",
            )

        if capability == CapabilityType.MEMORY:
            return Task(
                id=task_id,
                type="memory",
                action="Manage Memory",
                tool="memory",
                parameters=params,
                status="pending",
            )

        if capability in (CapabilityType.WEB, CapabilityType.VISION, CapabilityType.DEVICE):
            raise ValueError(
                f"Capability '{capability.value}' does not yet have a supported execution mapping in Phase 2.3"
            )

        raise ValueError(f"Unsupported capability: {capability}")

    def _extract_task_parameters(self, decision: Decision) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for key in (
            "query",
            "path",
            "tool_hint",
            "max_results",
            "timeout_seconds",
            "privacy_level",
            "confirmation_required",
            "ambiguity_reason",
            "collection",
            "author",
            "action",
            "memory_action",
            "key",
            "value",
            "user_id",
        ):
            if key in decision.routing_hints:
                params[key] = decision.routing_hints[key]

        if "action" not in params and "memory_action" in params:
            params["action"] = params["memory_action"]

        return params
