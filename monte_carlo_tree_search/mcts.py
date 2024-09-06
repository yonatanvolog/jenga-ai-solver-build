class MCTS:
    def __init__(self, env, exploration_weight=0.5):
        """
        Initializes the Monte Carlo Tree Search (MCTS) algorithm.

        Args:
            env (Environment): The Jenga game environment.
            exploration_weight (float, optional): The weight for the exploration term in UCB1. Defaults to 1.0.
        """
        self.env = env
        self.exploration_weight = exploration_weight

    def search(self, root):
        """
        Perform MCTS from the root node.

        Args:
            root (MCTSNode): The root node of the search tree.

        Returns:
            MCTSNode: The best child of the root node after running the simulations.
        """
        node = self._select(root)
        expanded_node = node.expand(self.env)
        if expanded_node:
            expanded_node.backpropagate()

        return root.best_child(exploration_weight=0.5)  # Best child according to the average reward

    def _select(self, node):
        """
        Selects a node to expand using the UCB formula.

        Args:
            node (MCTSNode): The node from which selection begins.

        Returns:
            MCTSNode: The node selected for expansion.
        """
        while node.children:
            node = node.best_child(self.exploration_weight)
        return node
