from abc import ABC, abstractmethod


class ExecutorInterface(ABC):

    @abstractmethod
    def execute(self, task):
        pass