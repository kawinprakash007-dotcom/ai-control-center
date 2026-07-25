from brain.intent_engine import IntentEngine


engine = IntentEngine()


intent = engine.detect(
    "Open calculator"
)

print(intent)