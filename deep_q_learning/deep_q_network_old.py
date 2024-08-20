import random
from collections import deque

import numpy as np
import torch
from torch import nn
import torch.nn.functional as activation_funcs


class DQN(nn.Module):
    def __init__(self, num_states, num_actions):
        super(DQN, self).__init__()

        # Network Layers:
        self.layer_1 = nn.Linear(num_states, 128)
        self.layer_2 = nn.Linear(128, 128)
        self.layer_3 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = activation_funcs.relu(self.layer_1(x))
        x = activation_funcs.relu(self.layer_2(x))
        return self.layer_3(x)


class ReplayMemory:
    def __init__(self, memory_size):
        self.memory = deque([], maxlen=memory_size)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)


class DQL:

    # Hyperparameters:
    learning_rate = 0.001  # alpha
    discount_factor = 0.9  # gamma
    network_sync_rate = 10  # number of steps before syncing the policy and target networks
    replay_memory_size = 1000  # size of the replay memory
    mini_batch_size = 32  # size of the training data set sampled from the replay memory

    loss_function = nn.MSELoss()
    optimizer = None

    def train(self, episodes):
        env = None  # create environment using Unity
        num_states = None
        num_actions = None
        epsilon = 1  # controls exploration-exploitation tradeoff
        memory = ReplayMemory(self.replay_memory_size)

        policy_dqn = DQN(num_states, num_actions)
        target_dqn = DQN(num_states, num_actions)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        reward = None
        rewards_per_episode = np.zeros(episodes)
        step_count = 0

        for i in range(episodes):
            state = None  # start from the initial state in the environment
            terminated = False  # True when Jenga falls
            truncated = False  # True when agent takes more than 100 actions

            while not terminated and not truncated:
                if random.random() < epsilon:
                    action = None  # get new action from env
                else:
                    with torch.no_grad():
                        action = policy_dqn(self.make_one_hot_encoding(state, num_states)).argmax().item()

                # execute action
                new_state, reward, terminated, truncated = env.step(action)

                # save experience into memory
                memory.append((state, action, new_state, reward, terminated))

                # update parameters
                state = new_state
                step_count += 1

            if reward == 1:
                rewards_per_episode[i] = 1

        if len(memory) > self.mini_batch_size and np.sum(rewards_per_episode) > 0:
            mini_batch = memory.sample(self.mini_batch_size)
            self.optimize(mini_batch, policy_dqn, target_dqn)
            epsilon = max(epsilon - 1/episodes, 0)

            if step_count > self.network_sync_rate:
                target_dqn.load_state_dict(policy_dqn.state_dict())
                global step_count
                step_count = 0

    @staticmethod
    def make_one_hot_encoding(state, num_states):
        dqn_input = torch.zeros(num_states)
        dqn_input[state] = 1
        return dqn_input
