from brain.intent_engine import IntentEngine
from brain.reasoner import Reasoner


intent_engine = IntentEngine()
reasoner = Reasoner()

intent = intent_engine.detect("Open calculator")

goal = reasoner.reason(intent)

print(intent)
print()
print(goal)