import math
import random

import utils


class MCTSNode:
    def __init__(self, state, parent=None, is_fallen=False, reward=0, action=None, taken_actions=set()):
        """
        Initializes a Monte Carlo Tree Search (MCTS) node.

        Args:
            state (torch.Tensor): The current state of the Jenga game represented as a tensor.
            parent (MCTSNode, optional): The parent node in the tree. Defaults to None.
            is_fallen (bool, optional): A flag indicating if the tower has fallen. Defaults to False.
            reward (float, optional): The reward obtained from taking the action that led to this state. Defaults to 0.
            action (tuple, optional): The action that was taken to reach this state, represented as (level, color).
                                      Defaults to None.
            taken_actions (set, optional): A set of actions that have already been taken in the game up to this point.
                                           Defaults to an empty set.
        """
        self.state = state  # The state of the Jenga game
        self.parent = parent  # The parent node in the tree
        self.children = []  # List of child nodes
        self.visits = 0  # Number of times the node has been visited
        self.action = action  # The action that led to this state
        self.is_fallen = is_fallen  # Is the tower fallen (that is, is the game terminated)
        self.reward = reward  # Reward of taking the current action
        self.taken_actions = taken_actions  # All the actions that were taken including this action
        self.possible_actions = utils.get_possible_actions(taken_actions)  # Actions possible to make from this node

    def expand(self, env):
        """
        Expands the current node by generating all possible child nodes.

        This function iterates over all possible actions that can be taken from the current state, simulates each
        action, and creates a corresponding child node. The child nodes are added to the list of this node's children.
        If the tower has already fallen (`is_fallen` is True), no expansion occurs.

        Args:
            env (Environment): The Jenga game environment that provides methods for interacting with the game,
                               such as taking actions, reverting steps, and getting the current state.

        Returns:
            None
        """
        if self.is_fallen or self.children:
            # Stop expansion if the node has already expanded or the tower has fallen
            return
        previous_stability = env.get_average_max_tilt_angle()
        for action in random.sample(self.possible_actions, min(len(self.possible_actions), 10)):
            print(f"Trying action {action}")
            screenshot_filename, is_fallen = env.step(action)
            current_stability = env.get_average_max_tilt_angle()
            next_state = utils.get_state_from_image(screenshot_filename)

            env.revert_step()

            reward = utils.calculate_reward(action, is_fallen, previous_stability, current_stability)
            child_node = MCTSNode(next_state, self, is_fallen, reward, action, self.taken_actions.union({action}))
            self.children.append(child_node)

    def backpropagate(self):
        """
        Backpropagates the reward from a leaf node up to the root, updating the visits and value of each node.

        The reward stored in each node is propagated up the tree, updating the node's value and visit count.

        Returns:
            None
        """
        # If this node has a parent, continue backpropagation
        if self.parent:
            self.parent.backpropagate()

    def ucb(self, exploration_weight=0.5):
        """
        Calculate the Upper Confidence Bound (UCB1) for this node.

        Args:
            exploration_weight (float): The constant controlling exploration. Higher values promote exploration.

        Returns:
            float: The UCB value for the node.
        """
        if self.visits == 0:
            return float('inf')  # Ensure that unvisited nodes are favored
        exploitation_value = self.reward
        exploration_value = exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation_value + exploration_value

    def best_child(self, exploration_weight=1.0):
        """
        Selects the child with the highest UCB1 value.

        Args:
            exploration_weight (float): The constant controlling exploration. Higher values promote exploration.

        Returns:
            MCTSNode: The child with the highest UCB1 value.
        """
        return max(self.children, key=lambda child: child.ucb(exploration_weight))
