from brain.goal_manager import GoalManager
from core.models.goal import Goal


manager = GoalManager()

manager.add_goal(
    Goal(goal="Launch Calculator")
)

manager.add_goal(
    Goal(goal="Launch Chrome")
)

manager.add_goal(
    Goal(goal="Launch VSCode")
)

print()

print("Current Goal")

print(manager.current_goal())

print()

print("Pending")

print(manager.pending_goals())

print()

print("Complete")

manager.complete_goal()

print()

print("Pending")

print(manager.pending_goals())