import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

import utils
from hierarchical_deep_q_learning.deep_q_network import DQN, ReplayMemory


class HierarchicalSARSAAgent:
    """
    Hierarchical SARSA agent for solving a Jenga game.

    This agent uses two separate Q-value approximators: one for determining the level of the block to remove,
    and another for determining the color of the block to remove. The agent employs an epsilon-greedy policy
    and updates its Q-values using SARSA.
    """

    def __init__(self, input_shape, num_actions_level_1, num_actions_level_2,
                 lr=1e-4, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1,
                 epsilon_decay=5000):
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

        # Q-networks for level and color selection
        self.policy_net_level_1 = DQN(input_shape, num_actions_level_1)
        self.policy_net_level_2 = DQN(input_shape, num_actions_level_2)
        self.target_net_level_1 = DQN(input_shape, num_actions_level_1)
        self.target_net_level_2 = DQN(input_shape, num_actions_level_2)

        # Set target networks to evaluation mode
        self.target_net_level_1.eval()
        self.target_net_level_2.eval()

        # Optimizers for both Q-networks
        self.optimizer_level_1 = optim.Adam(self.policy_net_level_1.parameters(), lr=lr)
        self.optimizer_level_2 = optim.Adam(self.policy_net_level_2.parameters(), lr=lr)

        # Replay memory
        self.memory = ReplayMemory(10000)

    def select_action(self, state, taken_actions, if_allow_exploration=True):
        """
        Select an action based on the current state using an epsilon-greedy policy.
        """
        self.steps_done += 1
        self.epsilon = self.epsilon_end + (self.epsilon - self.epsilon_end) * np.exp(
            -1. * self.steps_done / self.epsilon_decay)

        possible_actions = utils.get_possible_actions(taken_actions)
        if len(possible_actions) == 0:
            return None

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
        Optimize the Q-networks based on a batch of experiences from replay memory using SARSA.
        """
        if len(self.memory) < batch_size:
            return

        # Sample a batch of transitions from replay memory
        transitions = self.memory.sample(batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_next_action, batch_done = zip(*transitions)

        batch_state = torch.cat(batch_state)
        batch_reward = torch.tensor(batch_reward)
        batch_next_state = torch.cat(batch_next_state)
        batch_done = torch.tensor(batch_done, dtype=torch.float32)

        batch_action_level = torch.tensor([action[0] for action in batch_action]).unsqueeze(1)
        batch_action_color = torch.tensor([action[1] for action in batch_action]).unsqueeze(1)

        current_q_values_level_1 = self.policy_net_level_1(batch_state).gather(1, batch_action_level)
        current_q_values_level_2 = self.policy_net_level_2(batch_state).gather(1, batch_action_color)

        # SARSA: Use the Q-value of the next action, rather than the max Q-value
        batch_next_action_level = torch.tensor([action[0] for action in batch_next_action]).unsqueeze(1)
        batch_next_action_color = torch.tensor([action[1] for action in batch_next_action]).unsqueeze(1)

        next_q_values_level_1 = self.policy_net_level_1(batch_next_state).gather(1, batch_next_action_level).detach()
        next_q_values_level_2 = self.policy_net_level_2(batch_next_state).gather(1, batch_next_action_color).detach()

        expected_q_values_level_1 = (next_q_values_level_1 * self.gamma * (1 - batch_done)) + batch_reward
        expected_q_values_level_2 = (next_q_values_level_2 * self.gamma * (1 - batch_done)) + batch_reward

        # Compute loss
        loss_level_1 = F.mse_loss(current_q_values_level_1, expected_q_values_level_1.unsqueeze(1))
        loss_level_2 = F.mse_loss(current_q_values_level_2, expected_q_values_level_2.unsqueeze(1))

        # Backpropagate and update networks
        self.optimizer_level_1.zero_grad()
        self.optimizer_level_2.zero_grad()
        loss_level_1.backward()
        loss_level_2.backward()
        self.optimizer_level_1.step()
        self.optimizer_level_2.step()
