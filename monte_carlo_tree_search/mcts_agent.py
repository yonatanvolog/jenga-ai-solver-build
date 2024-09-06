import os
import pickle
from environment.environment import Environment
from mcts import MCTS
from monte_carlo_tree_search.node import MCTSNode


class MCTSAgent:
    def __init__(self, env, exploration_weight=0.5, tree_save_path="mcts_tree.pkl"):
        """
        Initializes the MCTSAgent with the given environment and parameters.

        Args:
            env (Environment): The Jenga game environment.
            exploration_weight (float, optional): The weight for the exploration term in UCB1. Defaults to 1.0.
            tree_save_path (str, optional): Path to save or load the MCTS tree. Defaults to "mcts_tree.pkl".
        """
        self.env = env
        self.exploration_weight = exploration_weight
        self.tree_save_path = tree_save_path
        self.root = None

        # # Load the existing tree if it exists
        # if os.path.exists(self.tree_save_path):
        #     self.load_tree()

    def save_tree(self):
        """
        Saves the current MCTS tree to a file.
        """
        with open(self.tree_save_path, 'wb') as f:
            pickle.dump(self.root, f)
        print(f"Tree saved to {self.tree_save_path}")

    def load_tree(self):
        """
        Loads the MCTS tree from a file.
        """
        with open(self.tree_save_path, 'rb') as f:
            self.root = pickle.load(f)
        print(f"Tree loaded from {self.tree_save_path}")

    def select_action(self, state, taken_actions):
        """
        Selects an action based on the current state using the MCTS strategy.

        Args:
            state (torch.Tensor): The current state of the environment.
            taken_actions (Set[Tuple[int, int]]): Already performed actions.

        Returns:
            tuple: A tuple containing the selected level and color of the block to remove.
        """
        # If no existing tree, create a new root node
        if not self.root:
            self.root = MCTSNode(state, taken_actions=taken_actions)

        mcts = MCTS(self.env, exploration_weight=self.exploration_weight)
        best_child = mcts.search(self.root)
        self.root = best_child  # Set the best child as the new root
        return best_child.action  # Return the action that led to the best child
