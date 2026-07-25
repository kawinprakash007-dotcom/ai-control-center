from core.models.result import Result


class Executor:

    def execute(self, tool_function):

        try:

            output = tool_function()

            return Result(

                success=True,

                message="Task completed.",

                output=str(output)

            )

        except Exception as e:

            return Result(

                success=False,

                message=str(e)

            )