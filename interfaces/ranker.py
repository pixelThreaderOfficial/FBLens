from abc import ABC, abstractmethod


class Ranker(ABC):

    @abstractmethod
    def rank(self, query, candidates):
        pass