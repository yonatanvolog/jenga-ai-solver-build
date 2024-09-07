import math
import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    Deep Q-Network (DQN) model for reinforcement learning tasks.

    This model consists of three convolutional layers followed by two fully connected layers.
    It takes an input image (state) and outputs Q-values for each possible action.

    Args:
        input_shape (tuple): The shape of the input image (height, width).
        num_actions (int): The number of possible actions the agent can take.

    Methods:
        forward(x): Performs a forward pass through the network and returns Q-values.
        _feature_size(shape): Computes the size of the feature map after the convolutional layers.
    """

    def __init__(self, input_shape, num_actions):
        """
        Initializes the DQN model.

        Args:
            input_shape (tuple): The shape of the input image (height, width).
            num_actions (int): The number of possible actions the agent can take.
        """
        super(DQN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(8, 8), stride=(4, 4))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))

        # Fully connected layers
        self.fc1 = nn.Linear(self._feature_size(input_shape), 512)
        self.fc2 = nn.Linear(512, num_actions)

    def _feature_size(self, shape):
        """
        Computes the size of the feature map after passing through the convolutional layers.

        Args:
            shape (tuple): The shape of the input image (height, width).

        Returns:
            int: The size of the flattened feature map.
        """
        # Create a dummy tensor with the shape of the input
        x = torch.zeros(1, 1, *shape)  # Adding 1 for the channel dimension

        # Pass it through the convolutional layers to calculate the resulting feature map size
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Compute the product of the size excluding the batch dimension (index 0)
        return math.prod(x.size()[1:])

    def forward(self, x):
        """
        Performs a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor (batch_size, channels, height, width).

        Returns:
            torch.Tensor: The output tensor containing Q-values for each action.
        """
        # Apply convolutional layers with ReLU activations
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten the tensor before passing it to the fully connected layers
        x = x.view(x.size(0), -1)

        # Apply the fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # Output Q-values for each action


class ReplayMemory:
    """
    Replay memory for storing and sampling experiences in reinforcement learning.

    This class uses a deque to store experiences up to a fixed capacity. It allows for
    sampling random mini-batches of experiences to train the DQN.

    Args:
        capacity (int): The maximum number of experiences to store.
    """

    def __init__(self, capacity):
        """
        Initializes the replay memory.

        Args:
            capacity (int): The maximum number of experiences to store.
        """
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Adds a new experience to the replay memory.

        Args:
            state (torch.Tensor): The state observed by the agent.
            action (int): The action taken by the agent.
            reward (float): The reward received after taking the action.
            next_state (torch.Tensor): The next state observed by the agent.
            done (bool): Whether the episode has ended.
        """
        self.memory.append((state, action, reward, next_state, done))

    def sarsa_push(self, state, action, reward, next_state, next_action, done):
        """
        Adds a new experience to the replay memory for SARSA (State-Action-Reward-State-Action) updates.

        This function stores also the predicted next action chosen in the next state, and whether the episode has
        ended. This is specifically for the SARSA algorithm, which updates the Q-value using the action taken in the
        next state.

        Args:
            state (torch.Tensor): The current state observed by the agent.
            action (int): The action taken by the agent in the current state.
            reward (float): The reward received after taking the action.
            next_state (torch.Tensor): The next state observed after taking the action.
            next_action (int): The next action chosen by the agent in the next state.
            done (bool): A flag indicating whether the episode has ended (True if the episode is over, False otherwise).
        """
        self.memory.append((state, action, reward, next_state, next_action, done))

    def sample(self, batch_size):
        """
        Samples a random batch of experiences from the memory.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            list: A list of sampled experiences.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """
        Returns the current number of experiences stored in the memory.

        Returns:
            int: The number of experiences in the memory.
        """
        return len(self.memory)
