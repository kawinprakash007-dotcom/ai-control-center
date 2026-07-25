from core.models.plan import Plan
from core.models.task import Task


class TaskManager:

    def __init__(self):

        self.queue = []

    def load_plan(self, plan: Plan):

        self.queue = plan.steps.copy()

    def has_tasks(self):

        return len(self.queue) > 0

    def next_task(self):

        if not self.queue:
            return None

        task = self.queue.pop(0)

        task.status = "running"

        return task

    def pending_tasks(self):

        return len(self.queue)

    def clear(self):

        self.queue.clear()