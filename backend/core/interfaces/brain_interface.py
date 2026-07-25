from abc import ABC, abstractmethod


class BrainInterface(ABC):

    @abstractmethod
    def think(self, message):
        pass