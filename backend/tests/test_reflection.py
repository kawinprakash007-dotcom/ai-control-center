from brain.reflection import Reflection
from core.models.result import Result


reflection = Reflection()

result = Result(

    success=True,

    message="Calculator opened"

)

decision = reflection.evaluate(result)

print(decision)