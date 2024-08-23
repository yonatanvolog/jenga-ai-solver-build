from abc import ABC, abstractmethod
import random

from environment.environment import MAX_LEVEL, MAX_BLOCKS_IN_LEVEL


class Strategy(ABC):
    @abstractmethod
    def select_action(self, state, previous_action):
        pass


class RandomStrategy(Strategy):

    def select_action(self, state, previous_action=None):
        level = random.randrange(0, MAX_LEVEL)
        color = random.randrange(0, MAX_BLOCKS_IN_LEVEL)
        return level, color


class OptimisticStrategy(Strategy):

    def select_action(self, state, previous_action=None):
        if previous_action:
            previous_level = previous_action[0]
            min_level = min(MAX_LEVEL, previous_level + 1)
            level = random.randrange(min_level, MAX_LEVEL)
        else:
            level = random.randrange(0, MAX_LEVEL)
        color = random.randrange(0, MAX_BLOCKS_IN_LEVEL)
        return level, color


class PessimisticStrategy(Strategy):

    def select_action(self, state, previous_action=None):
        if previous_action:
            previous_level = previous_action[0]
            max_level = max(0, previous_level - 1)
            level = random.randrange(0, max_level)
        else:
            level = random.randrange(0, MAX_LEVEL)
        color = random.choice(MAX_BLOCKS_IN_LEVEL)
        return level, color
