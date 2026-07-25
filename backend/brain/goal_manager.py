from core.models.goal import Goal


class GoalManager:

    def __init__(self):

        self.goals = []

    def add_goal(self, goal: Goal):

        self.goals.append(goal)

    def current_goal(self):

        if self.goals:

            return self.goals[0]

        return None

    def complete_goal(self):

        if self.goals:

            return self.goals.pop(0)

        return None

    def pending_goals(self):

        return self.goals

    def clear(self):

        self.goals.clear()