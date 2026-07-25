from brain.intent_engine import IntentEngine
from brain.reasoner import Reasoner
from brain.planner import Planner


intent_engine = IntentEngine()
reasoner = Reasoner()
planner = Planner()


intent = intent_engine.detect(
    "Open calculator"
)

goal = reasoner.reason(intent)

plan = planner.create_plan(goal)

print()

print("Intent")

print(intent)

print()

print("Goal")

print(goal)

print()

print("Plan")

print(plan)

print()

for task in plan.steps:

    print(task)