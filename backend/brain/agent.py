from brain.context_engine import ContextEngine
from brain.intent_engine import IntentEngine
from brain.reasoner import Reasoner
from brain.goal_manager import GoalManager
from brain.planner import Planner
from brain.task_manager import TaskManager
from brain.router import Router
from brain.reflection import Reflection

from memory.history import add_message


class Agent:

    def __init__(self):

        self.context = ContextEngine()
        self.intent_engine = IntentEngine()
        self.reasoner = Reasoner()
        self.goal_manager = GoalManager()
        self.planner = Planner()
        self.task_manager = TaskManager()
        self.router = Router()
        self.reflection = Reflection()

    def think(self, message):

        print("\n==============================")
        print("      JARVIS PIPELINE")
        print("==============================")

        print(f"\n[USER] {message}")

        # --------------------------------
        # Context
        # --------------------------------

        self.context.update_user_message(message)
        add_message("user", message)

        print("\n[✓] Context Updated")

        # --------------------------------
        # Intent
        # --------------------------------

        intent = self.intent_engine.detect(message)

        print("\n[Intent]")
        print(intent)

        # --------------------------------
        # Reasoning
        # --------------------------------

        goal = self.reasoner.reason(intent)

        self.goal_manager.add_goal(goal)

        self.context.set_goal(goal.goal)

        print("\n[Goal]")
        print(goal)

        # --------------------------------
        # Planning
        # --------------------------------

        plan = self.planner.create_plan(goal)

        print("\n[Plan]")
        print(plan)

        # --------------------------------
        # Load Tasks
        # --------------------------------

        self.task_manager.load_plan(plan)

        print(
            f"\nLoaded {self.task_manager.pending_tasks()} task(s)"
        )

        final_response = "Done."

        # --------------------------------
        # Execute Tasks
        # --------------------------------

        while self.task_manager.has_tasks():

            task = self.task_manager.next_task()

            print("\n----------------------------")
            print("Current Task")
            print(task)

            self.context.set_current_task(task.action)

            if task.tool:
                self.context.set_last_tool(task.tool)

            result = self.router.route(task)

            print("\n========== RESULT ==========")
            print(result)
            print("Message :", result.message)
            print("Output  :", result.output)
            print("============================")

            decision = self.reflection.evaluate(result)

            print("\nReflection")
            print(decision)

            # --------------------------------
            # Return the ACTUAL OUTPUT
            # --------------------------------

            if result.output:

                final_response = result.output

            else:

                final_response = result.message

            if not decision.continue_execution:

                print("\nStopping execution.")

                break

        # --------------------------------
        # Goal Completed
        # --------------------------------

        completed = self.goal_manager.complete_goal()

        print("\nCompleted Goal")
        print(completed)

        # --------------------------------
        # Save AI Response
        # --------------------------------

        self.context.update_ai_response(final_response)

        add_message(
            "assistant",
            final_response
        )

        print("\nPipeline Finished")
        print("==============================\n")

        return final_response