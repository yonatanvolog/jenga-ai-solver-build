from abc import ABC, abstractmethod
import random

from dev_environment.environment import MAX_LEVEL, MAX_BLOCKS_IN_LEVEL


class Strategy(ABC):
    """
    An abstract base class for defining different strategies for selecting actions in the Jenga game.

    Subclasses must implement the `select_action` method, which determines how a move (level and color) is selected
    based on the current state and the previous action.
    """

    @abstractmethod
    def select_action(self, state, previous_action):
        """
        Selects an action (level and color) based on the current state and the previous action.

        Args:
            state (any): The current state of the environment (not used in this implementation but can be in others).
            previous_action (tuple or None): The previous action taken, as a tuple (level, color).

        Returns:
            tuple: A tuple containing the selected level and color (level, color).
        """
        pass


class RandomStrategy(Strategy):
    """
    A strategy that selects actions randomly, without considering the previous action.

    The RandomStrategy chooses a random level and a random color for each move.
    """

    def select_action(self, state, previous_action=None):
        """
        Selects a random action (level and color) without considering the previous action.

        Args:
            state (any): The current state of the environment (not used).
            previous_action (tuple or None): The previous action taken (not used).

        Returns:
            tuple: A tuple containing a randomly selected level and color (level, color).
        """
        level = random.randrange(0, MAX_LEVEL)
        color = random.randrange(0, MAX_BLOCKS_IN_LEVEL)
        return level, color


class OptimisticStrategy(Strategy):
    """
    A strategy that favors selecting actions from levels higher than the previous move.

    The OptimisticStrategy attempts to select a block from a level above the previous move's level. If there
    is no previous action, it selects a random level and color.
    """

    def select_action(self, state, previous_action=None):
        """
        Selects an action with a preference for levels higher than the previous move's level.

        Args:
            state (any): The current state of the environment (not used).
            previous_action (tuple or None): The previous action taken, as a tuple (level, color).

        Returns:
            tuple: A tuple containing the selected level and color (level, color).
        """
        if previous_action:
            previous_level = previous_action[0]
            min_level = min(previous_level + 1, MAX_LEVEL)
            try:
                level = random.randrange(min_level, MAX_LEVEL)
            except:
                level = random.randrange(0, MAX_LEVEL)
        else:
            level = random.randrange(0, MAX_LEVEL)
        color = random.randrange(0, MAX_BLOCKS_IN_LEVEL)
        return level, color


class PessimisticStrategy(Strategy):
    """
    A strategy that favors selecting actions from levels lower than the previous move.

    The PessimisticStrategy attempts to select a block from a level below the previous move's level. If there
    is no previous action, it selects a random level and color.
    """

    def select_action(self, state, previous_action=None):
        """
        Selects an action with a preference for levels lower than the previous move's level.

        Args:
            state (any): The current state of the environment (not used).
            previous_action (tuple or None): The previous action taken, as a tuple (level, color).

        Returns:
            tuple: A tuple containing the selected level and color (level, color).
        """
        if previous_action:
            previous_level = previous_action[0]
            max_level = max(0, previous_level - 1)
            try:
                level = random.randrange(0, max_level + 1)
            except:
                level = random.randrange(0, MAX_LEVEL)
        else:
            level = random.randrange(0, MAX_LEVEL)
        color = random.randrange(0, MAX_BLOCKS_IN_LEVEL)
        return level, color
