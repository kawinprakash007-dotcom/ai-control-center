from core.models.plan import Plan
from core.models.task import Task
from core.models.goal import Goal
from core.constants.task_types import TaskType


class Planner:

    def create_plan(self, goal: Goal) -> Plan:

        plan = Plan(
            goal=goal.goal,
            confidence=goal.confidence
        )

        # -----------------------------
        # Tool Goals
        # -----------------------------

        if goal.goal.startswith("Launch"):

            tool = goal.goal.replace("Launch ", "")

            plan.steps.append(

                Task(

                    id=1,

                    type=TaskType.TOOL.value,

                    action=f"Open {tool}",

                    tool=tool.lower()

                )

            )

            return plan

        # -----------------------------
        # Memory Goals
        # -----------------------------

        if goal.goal == "Access Memory":

            plan.steps.append(

                Task(

                    id=1,

                    type=TaskType.MEMORY.value,

                    action="Access Memory",

                    tool="memory"
                )

            )

            return plan

        # -----------------------------
        # Default Chat
        # -----------------------------

        plan.steps.append(

            Task(

                id=1,

                type=TaskType.CHAT.value,

                action="Respond to User",

                tool="chat"
            )

        )

        return plan