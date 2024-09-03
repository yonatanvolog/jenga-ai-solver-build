from environment.environment import Environment
from mcts import MCTS
from monte_carlo_tree_search.node import MCTSNode


class MCTSAgent:
    def __init__(self, env, num_simulations=1000, exploration_weight=0.5):
        """
        Initializes the MCTSAgent with the given environment and parameters.

        Args:
            env (Environment): The Jenga game environment.
            num_simulations (int, optional): The number of simulations to run for each action selection. Defaults to 1000.
            exploration_weight (float, optional): The weight for the exploration term in UCB1. Defaults to 1.0.
        """
        self.env = env
        self.mcts = MCTS(env, num_simulations=num_simulations, exploration_weight=exploration_weight)

    def select_action(self, state, taken_actions):
        """
        Selects an action based on the current state using the MCTS strategy.

        Args:
            state (torch.Tensor): The current state of the environment.
            taken_actions (Set[Tuple[int, int]]): Already performed actions.

        Returns:
            tuple: A tuple containing the selected level and color of the block to remove.
        """
        root = MCTSNode(state, taken_actions=taken_actions)
        best_child = self.mcts.search(root)
        return best_child.action  # Return the action that led to the best child
