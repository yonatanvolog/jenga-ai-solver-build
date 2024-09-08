import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

import utils
from deep_q_network.deep_q_network import DQN, ReplayMemory
from environment.environment import SCREENSHOT_SHAPE, MAX_LEVEL, MAX_BLOCKS_IN_LEVEL


class HierarchicalDQNAgent:
    """
    Hierarchical Deep Q-Network (DQN) agent for solving a Jenga game.

    This agent uses two separate DQNs: one for determining the level of the block to remove,
    and another for determining the color of the block to remove. The agent employs an
    epsilon-greedy policy to balance exploration and exploitation during learning.

    Args:
        input_shape (tuple): The shape of the input state (height, width).
        num_actions_level_1 (int): The number of possible actions for selecting the level.
        num_actions_level_2 (int): The number of possible actions for selecting the color.
        lr (float): Learning rate for the optimizer.
        gamma (float): Discount factor for future rewards.
        epsilon_start (float): Initial value for epsilon in the epsilon-greedy policy.
        epsilon_end (float): Minimum value for epsilon after decay.
        epsilon_decay (int): The rate at which epsilon decays.

    Methods:
        select_action(state): Selects an action based on the current state using the epsilon-greedy policy.
        optimize_model(batch_size): Optimizes the policy networks based on a batch of experiences from replay memory.
        update_target_net(): Updates the target networks with the current weights of the policy networks.
    """

    def __init__(self, input_shape=SCREENSHOT_SHAPE, num_actions_level_1=MAX_LEVEL,
                 num_actions_level_2=MAX_BLOCKS_IN_LEVEL, lr=1e-4, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1,
                 epsilon_decay=5000):
        """
        Initializes the HierarchicalDQNAgent.

        Args:
            input_shape (tuple): The shape of the input state (height, width).
            num_actions_level_1 (int): The number of possible actions for selecting the level.
            num_actions_level_2 (int): The number of possible actions for selecting the color.
            lr (float): Learning rate for the optimizer.
            gamma (float): Discount factor for future rewards.
            epsilon_start (float): Initial value for epsilon in the epsilon-greedy policy.
            epsilon_end (float): Minimum value for epsilon after decay.
            epsilon_decay (int): The rate at which epsilon decays.
        """
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

        # DQNs for level and color selection
        self.policy_net_level_1 = DQN(input_shape, num_actions_level_1)
        self.policy_net_level_2 = DQN(input_shape, num_actions_level_2)
        self.target_net_level_1 = DQN(input_shape, num_actions_level_1)
        self.target_net_level_2 = DQN(input_shape, num_actions_level_2)

        # Load policy network weights into target networks
        self.target_net_level_1.load_state_dict(self.policy_net_level_1.state_dict())
        self.target_net_level_2.load_state_dict(self.policy_net_level_2.state_dict())

        # Set target networks to evaluation mode
        self.target_net_level_1.eval()
        self.target_net_level_2.eval()

        # Optimizers for the policy networks
        self.optimizer_level_1 = optim.Adam(self.policy_net_level_1.parameters(), lr=lr)
        self.optimizer_level_2 = optim.Adam(self.policy_net_level_2.parameters(), lr=lr)

        # Replay memory to store experiences
        self.memory = ReplayMemory(10000)

    def save_model(self, level_1_path="level_1.pth", level_2_path="level_2.pth"):
        """
        Save the weights of the policy networks to files.

        Args:
            level_1_path (str): Path to save the level 1 network weights.
            level_2_path (str): Path to save the level 2 network weights.
        """
        torch.save(self.policy_net_level_1.state_dict(), level_1_path)
        torch.save(self.policy_net_level_2.state_dict(), level_2_path)
        print(f"Model saved to {level_1_path} and {level_2_path}")

    def load_model(self, level_1_path="level_1.pth", level_2_path="level_2.pth"):
        """
        Load the weights of the policy networks from files.

        Args:
            level_1_path (str): Path to load the level 1 network weights from.
            level_2_path (str): Path to load the level 2 network weights from.
        """
        self.policy_net_level_1.load_state_dict(torch.load(level_1_path, weights_only=True))
        self.policy_net_level_2.load_state_dict(torch.load(level_2_path, weights_only=True))
        self.target_net_level_1.load_state_dict(self.policy_net_level_1.state_dict())
        self.target_net_level_2.load_state_dict(self.policy_net_level_2.state_dict())
        print(f"Model loaded from {level_1_path} and {level_2_path}")

    def clone_model(self):
        """
        Clones the current policy networks to create an adversary.

        Returns:
            HierarchicalDQNAgent: A new instance of HierarchicalDQNAgent initialized with the same weights.
        """
        adversary = HierarchicalDQNAgent()
        adversary.policy_net_level_1.load_state_dict(self.policy_net_level_1.state_dict())
        adversary.policy_net_level_2.load_state_dict(self.policy_net_level_2.state_dict())
        adversary.update_target_net()  # Synchronize target networks
        return adversary

    def select_action(self, state, taken_actions, if_allow_exploration=True):
        """
        Selects an action based on the current state using the epsilon-greedy policy.

        Args:
            state (torch.Tensor): The current state of the environment.
            taken_actions (Set[Tuple[int, int]]): Already performed actions.
            if_allow_exploration (bool): Specifies if exploration is allowed. Defaults to True.

        Returns:
            tuple: A tuple containing the selected level and color of the block to remove.
        """
        self.steps_done += 1
        # Update epsilon for exploration-exploitation trade-off
        self.epsilon = self.epsilon_end + (self.epsilon - self.epsilon_end) * \
            np.exp(-1. * self.steps_done / self.epsilon_decay)

        possible_actions = utils.get_possible_actions(taken_actions)
        if len(possible_actions) == 0:  # If no free actions, return None
            return

        # Choose action based on epsilon-greedy policy
        if not if_allow_exploration or random.random() > self.epsilon:
            # Exploitation: Select the best action that hasn't been taken yet
            best_q_value = float('-inf')
            best_action = random.choice(possible_actions)
            with torch.no_grad():
                for action in possible_actions:
                    level, color = action
                    q_value_level = self.policy_net_level_1(state)[0, level].item()
                    q_value_color = self.policy_net_level_2(state)[0, color].item()
                    if q_value_level + q_value_color > best_q_value:
                        best_q_value = q_value_level + q_value_color
                        best_action = action
                print(f"Exploiting: Selected action: {best_action}")
        else:
            best_action = random.choice(possible_actions)
            print(f"Exploring: Selected action {best_action}")

        taken_actions.add(best_action)  # Record the action as taken
        return best_action

    def optimize_model(self, batch_size):
        """
        Optimizes the policy networks based on a batch of experiences from replay memory using the DQN approach.

        Args:
            batch_size (int): The number of experiences to sample from replay memory for training.
        """
        if len(self.memory) < batch_size:
            return

        # Sample a batch of transitions from replay memory
        transitions = self.memory.sample(batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

        # Convert batches to tensors
        batch_state = torch.cat(batch_state)  # Shape: (batch_size, input_shape)
        batch_next_state = torch.cat(batch_next_state)  # Shape: (batch_size, input_shape)

        # Convert rewards and done signals to tensors of shape (batch_size, 1)
        batch_reward = torch.tensor(batch_reward, dtype=torch.float32).view(batch_size, 1)
        batch_done = torch.tensor(batch_done, dtype=torch.float32).view(batch_size, 1)

        # Split actions into separate tensors for level and color, and reshape them
        batch_action_level = torch.tensor([action[0] for action in batch_action]).unsqueeze(1)
        batch_action_color = torch.tensor([action[1] for action in batch_action]).unsqueeze(1)

        # Get the current Q-values for the actions that were taken
        current_q_values_level_1 = self.policy_net_level_1(batch_state).gather(1, batch_action_level)
        current_q_values_level_2 = self.policy_net_level_2(batch_state).gather(1, batch_action_color)

        # Compute the target Q-values using the target network and Bellman equation
        next_q_values_level_1 = self.target_net_level_1(batch_next_state).max(1)[0].detach().unsqueeze(1)
        next_q_values_level_2 = self.target_net_level_2(batch_next_state).max(1)[0].detach().unsqueeze(1)

        # Bellman equation: expected Q-values for both level and color selections
        expected_q_values_level_1 = (next_q_values_level_1 * self.gamma * (1 - batch_done)) + batch_reward
        expected_q_values_level_2 = (next_q_values_level_2 * self.gamma * (1 - batch_done)) + batch_reward

        # Compute the loss between current and expected Q-values
        loss_level_1 = F.mse_loss(current_q_values_level_1, expected_q_values_level_1)
        loss_level_2 = F.mse_loss(current_q_values_level_2, expected_q_values_level_2)

        # Zero gradients, backpropagate the loss, and update the network weights
        self.optimizer_level_1.zero_grad()
        self.optimizer_level_2.zero_grad()
        loss_level_1.backward()
        loss_level_2.backward()
        self.optimizer_level_1.step()
        self.optimizer_level_2.step()

    def update_target_net(self):
        """
        Updates the target networks with the current weights of the policy networks.
        """
        self.target_net_level_1.load_state_dict(self.policy_net_level_1.state_dict())
        self.target_net_level_2.load_state_dict(self.policy_net_level_2.state_dict())
