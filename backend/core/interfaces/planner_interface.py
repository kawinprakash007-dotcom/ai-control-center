from abc import ABC, abstractmethod


class PlannerInterface(ABC):

    @abstractmethod
    def create_plan(self, goal):
        pass