import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from deep_q_learning.deep_q_network import DQN, ReplayMemory


INT_TO_COLOR = {0: "y", 1: "b", 2: "g"}


class HierarchicalDQNAgent:
    def __init__(self, input_shape, num_actions_level_1, num_actions_level_2,
                 lr=1e-4, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1,
                 epsilon_decay=1000):
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

        self.policy_net_level_1 = DQN(input_shape, num_actions_level_1)  # a DQN for determining the number of level
        self.policy_net_level_2 = DQN(input_shape, num_actions_level_2)  # a DQN for determining the color
        self.target_net_level_1 = DQN(input_shape, num_actions_level_1)
        self.target_net_level_2 = DQN(input_shape, num_actions_level_2)
        self.target_net_level_1.load_state_dict(self.policy_net_level_1.state_dict())
        self.target_net_level_2.load_state_dict(self.policy_net_level_2.state_dict())
        self.target_net_level_1.eval()
        self.target_net_level_2.eval()

        self.optimizer_level_1 = optim.Adam(self.policy_net_level_1.parameters(), lr=lr)
        self.optimizer_level_2 = optim.Adam(self.policy_net_level_2.parameters(), lr=lr)
        self.memory = ReplayMemory(10000)

    def select_action(self, state):
        self.steps_done += 1
        self.epsilon = self.epsilon_end + (self.epsilon - self.epsilon_end) * \
            np.exp(-1. * self.steps_done / self.epsilon_decay)

        if random.random() > self.epsilon:
            with torch.no_grad():
                return self.policy_net_level_1(state).argmax(dim=1).item(), \
                       INT_TO_COLOR[self.policy_net_level_2(state).argmax(dim=1).item()]
        else:
            return random.randrange(0, 9), INT_TO_COLOR[random.randrange(0, 2)]

    def optimize_model(self, batch_size):
        if len(self.memory) < batch_size:
            return

        transitions = self.memory.sample(batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

        batch_state = torch.cat(batch_state)
        batch_action = torch.tensor(batch_action).unsqueeze(1)
        batch_reward = torch.tensor(batch_reward)
        batch_next_state = torch.cat(batch_next_state)
        batch_done = torch.tensor(batch_done, dtype=torch.float32)

        current_q_values_level_1 = self.policy_net_level_1(batch_state).gather(1, batch_action)
        next_q_values_level_1 = self.target_net_level_1(batch_next_state).max(1)[0].detach()
        current_q_values_level_2 = self.policy_net_level_2(batch_state).gather(1, batch_action)
        next_q_values_level_2 = self.target_net_level_2(batch_next_state).max(1)[0].detach()
        expected_q_values_level_1 = (next_q_values_level_1 * self.gamma * (1 - batch_done)) + batch_reward
        expected_q_values_level_2 = (next_q_values_level_2 * self.gamma * (1 - batch_done)) + batch_reward

        loss_level_1 = F.mse_loss(current_q_values_level_1, expected_q_values_level_1.unsqueeze(1))
        loss_level_2 = F.mse_loss(current_q_values_level_2, expected_q_values_level_2.unsqueeze(1))

        self.optimizer_level_1.zero_grad()
        self.optimizer_level_2.zero_grad()
        loss_level_1.backward()
        loss_level_2.backward()
        self.optimizer_level_1.step()
        self.optimizer_level_2.step()

    def update_target_net(self):
        self.target_net_level_1.load_state_dict(self.policy_net_level_1.state_dict())
        self.target_net_level_2.load_state_dict(self.policy_net_level_2.state_dict())
