from deep_q_learning.strategy import RandomStrategy
from environment.environment import MAX_BLOCKS_IN_LEVEL, MAX_LEVEL


class Adversary:
    """
    Represents an adversary in the Jenga game that selects actions based on a specified strategy.

    The Adversary class uses a strategy to choose which block to remove in the Jenga game. The strategy can be
    customized, and the default strategy is random. The adversary checks whether an action has already been taken and
    attempts to select a new action if the chosen one is not valid.

    Args:
        strategy (Strategy): The strategy used by the adversary to select actions. Defaults to `RandomStrategy()`.

    Methods:
        select_action(state, taken_actions, previous_action): Selects an action based on the current state, avoiding any
                                                              actions that have already been taken.
    """

    def __init__(self, strategy=RandomStrategy()):
        """
        Initializes the Adversary with a specified strategy.

        Args:
            strategy (Strategy): The strategy used by the adversary to select actions. Defaults to `RandomStrategy()`.
        """
        self.strategy = strategy

    def select_action(self, state, taken_actions, previous_action):
        """
        Selects an action based on the current state, avoiding any actions that have already been taken.

        This method uses the adversary's strategy to choose an action, ensuring that it does not repeat actions that
        have already been taken. If all possible actions have been taken, the method returns `None`.

        Args:
            state (any): The current state of the environment (not used in this implementation but required by the
                         strategy).
            taken_actions (set): A set of actions that have already been taken in the current game.
            previous_action (tuple or None): The previous action taken by the agent, used to inform the strategy's next
                                             action.

        Returns:
            tuple or None: The selected action as a tuple (level, color) if a valid action is available, or `None` if
                           all actions have been exhausted.

        Example:
            action = adversary.select_action(state, taken_actions, previous_action)
            if action is None:
                print("No more valid actions available.")
        """
        if len(taken_actions) == MAX_LEVEL * MAX_BLOCKS_IN_LEVEL:
            return None

        possible_action = self.strategy.select_action(state, previous_action)
        if possible_action in taken_actions:
            possible_action = self.strategy.select_action(state)

        taken_actions.add(possible_action)
        return possible_action
