from brain.intent_engine import IntentEngine
from brain.reasoner import Reasoner
from brain.planner import Planner
from brain.task_manager import TaskManager
from brain.router import Router


intent_engine = IntentEngine()
reasoner = Reasoner()
planner = Planner()
manager = TaskManager()
router = Router()


intent = intent_engine.detect(
    "Open calculator"
)

goal = reasoner.reason(intent)

plan = planner.create_plan(goal)

manager.load_plan(plan)

while manager.has_tasks():

    task = manager.next_task()

    print(task)

    result = router.route(task)

    print(result)

    print("-" * 40)