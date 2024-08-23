from deep_q_learning.strategy import RandomStrategy
from environment.environment import MAX_BLOCKS_IN_LEVEL, MAX_LEVEL


class Adversary:
    def __init__(self, strategy=RandomStrategy()):
        self.strategy = strategy

    def select_action(self, state, taken_actions, previous_action):
        if taken_actions == MAX_LEVEL * MAX_BLOCKS_IN_LEVEL:
            return None
        possible_action = self.strategy.select_action(state, previous_action)
        if possible_action in taken_actions:
            possible_action = self.strategy.select_action(state)
        taken_actions.add(possible_action)
        return possible_action
