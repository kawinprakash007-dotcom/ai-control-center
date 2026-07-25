from core.models.result import Result
from core.models.reflection import ReflectionDecision


class Reflection:

    def evaluate(self, result: Result):

        if result.success:

            return ReflectionDecision(

                success=True,

                continue_execution=True,

                retry=False,

                remember=True,

                confidence=1.0,

                message="Execution successful."

            )

        return ReflectionDecision(

            success=False,

            continue_execution=False,

            retry=True,

            remember=False,

            confidence=0.5,

            message=result.message

        )