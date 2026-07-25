from core.models.goal import Goal
from core.models.intent import Intent


class Reasoner:

    def reason(self, intent: Intent) -> Goal:

        # Tool requests
        if intent.intent == "tool":

            return Goal(

                goal=f"Launch {intent.tool}",

                priority="normal",

                status="pending",

                confidence=intent.confidence,

                reason=intent.reason

            )

        # Memory requests
        if intent.intent == "memory":

            return Goal(

                goal="Access Memory",

                priority="high",

                status="pending",

                confidence=intent.confidence,

                reason=intent.reason

            )

        # Automation
        if intent.intent == "automation":

            return Goal(

                goal="Create Automation",

                priority="normal",

                status="pending",

                confidence=intent.confidence,

                reason=intent.reason

            )

        # Vision
        if intent.intent == "vision":

            return Goal(

                goal="Analyze Image",

                priority="normal",

                status="pending",

                confidence=intent.confidence,

                reason=intent.reason

            )

        # Default Chat
        return Goal(

            goal="Respond to User",

            priority="normal",

            status="pending",

            confidence=intent.confidence,

            reason=intent.reason

        )