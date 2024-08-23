import itertools
import random

import torchvision.transforms as transforms
from PIL import Image

from deep_q_learning.adversary import Adversary
from deep_q_learning.deep_q_agent import HierarchicalDQNAgent
from deep_q_learning.strategy import RandomStrategy, PessimisticStrategy, OptimisticStrategy
from environment.environment import Environment


def load_image(filename):
    """
    Loads an image from the specified file.

    Args:
        filename (str): The path to the image file.

    Returns:
        PIL.Image.Image: The loaded image.
    """
    # Open the image file
    image = Image.open(filename)
    return image


def preprocess_image(image):
    """
    Preprocesses the input image by converting it to grayscale, resizing,
    converting to a tensor, and normalizing it.

    Args:
        image (PIL.Image.Image): The image to preprocess.

    Returns:
        torch.Tensor: The preprocessed image as a tensor with a batch dimension.
    """
    preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((128, 64)),  # Resize to 128x64 pixels
        transforms.ToTensor(),        # Convert to tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])
    return preprocess(image).unsqueeze(0)  # Add batch dimension


def calculate_reward(action, is_fallen):
    """
    Calculates the reward for the agent's action.

    Args:
        action (tuple): The action taken by the agent, including the level and color.
        is_fallen (bool): Whether the tower fell after the action.

    Returns:
        int: The calculated reward, including a penalty if the tower fell.
    """
    reward = action[0] + (1 if action[1] == "b" else 0)  # Reward based on the level of the block removed

    if is_fallen:
        reward = -50  # Apply penalty if the tower falls

    return reward


def update_epsilon(agent, efficiency_threshold, move_count):
    """
        Adjusts the epsilon value for the agent's epsilon-greedy policy based on the efficiency of the current strategy.

        This function modifies the agent's epsilon, which controls the balance between exploration and exploitation.
        If the number of moves made before the tower falls is below the efficiency threshold, epsilon is increased
        to promote more exploration in the next episode. Conversely, if the move count meets or exceeds the threshold,
        epsilon is decreased to encourage exploitation of the current strategy.

        Args:
            agent (HierarchicalDQNAgent): The agent whose epsilon value is being adjusted.
            efficiency_threshold (int): The minimum number of moves before the tower falls to consider the strategy
                                        efficient.
            move_count (int): The number of moves made in the current episode before the tower fell.

        Side Effects:
            - Modifies the `epsilon` attribute of the agent, increasing it if the move count is below the efficiency
                threshold,
              or decreasing it if the move count meets or exceeds the threshold.

        Example:
            update_epsilon(agent, efficiency_threshold=10, move_count=8)
            # If move_count is 8 and efficiency_threshold is 10, epsilon will be increased to encourage exploration.

        Prints:
            - A message indicating whether exploration was increased or decreased, along with the new epsilon value.
    """
    if move_count < efficiency_threshold:
        agent.epsilon = min(1.0, agent.epsilon * 1.1)  # Increase epsilon to promote exploration
        print(f"Increased exploration: epsilon = {agent.epsilon}")
    else:
        agent.epsilon = max(agent.epsilon_end, agent.epsilon * 0.9)  # Decrease epsilon to promote exploitation
        print(f"Decreased exploration: epsilon = {agent.epsilon}")


def training_loop(if_load_weights=True, if_training_against_adversary=False, strategy=RandomStrategy(),
                  efficiency_threshold=10):
    """
    Runs the training loop for the HierarchicalDQNAgent in a Jenga environment.

    The agent interacts with the environment over a series of episodes. If `adversary_training` is True,
    the agent will train against an adversary initialized with the agent's weights from the last phase.

    Args:
        if_load_weights (bool): Whether to load pre-existing model weights if they exist or start from scratch.
        if_training_against_adversary (bool): Whether to train against a DNN adversary.
        strategy (Strategy): Strategy for the adversary to take.
        efficiency_threshold (int): The minimum number of moves before the tower falls to consider the strategy
                                    efficient.
    """
    print("Starting a new training loop")

    num_episodes = 50
    batch_size = 10
    target_update = 10

    # Initialize the agent and environment
    agent = HierarchicalDQNAgent(input_shape=(128, 64), num_actions_level_1=12, num_actions_level_2=3)
    env = Environment()
    env.set_timescale(100)  # Speed up the simulation

    # Load model weights if they exist
    if if_load_weights:
        try:
            agent.load_model()
        except FileNotFoundError:
            print("No previous model found. Starting from scratch")

    # Create an adversary based on the agent's current model
    adversary = None
    if if_training_against_adversary:
        adversary = Adversary(strategy)
    if adversary:
        print("The agent is training against an adversary with random strategy")

    for episode in range(num_episodes):
        _run_episode(adversary, agent, batch_size, efficiency_threshold, env, episode, num_episodes, target_update)

    # Save model weights at the end of the training session
    agent.save_model()


def _run_episode(adversary, agent, batch_size, efficiency_threshold, env, episode, num_episodes, target_update):
    """
       Runs a single episode of training for the HierarchicalDQNAgent in the Jenga environment.

       During the episode, the agent (and optionally an adversary) interacts with the environment, selecting actions
       and optimizing its model based on the results of those actions. The episode continues until the Jenga tower
       falls, or no more valid actions are available.

       Args:
           adversary (Adversary or None): An optional adversary agent that may take turns with the main agent.
           agent (HierarchicalDQNAgent): The main agent being trained.
           batch_size (int): The number of experiences to sample from replay memory for training.
           efficiency_threshold (int): The minimum number of moves before the tower falls to consider the strategy
                                       efficient.
           env (Environment): The environment representing the Jenga game.
           episode (int): The current episode number.
           num_episodes (int): The total number of episodes in the training session.
           target_update (int): The frequency (in episodes) at which the target network is updated.

       Side Effects:
           - Resets the environment and the `taken_actions` set of the agent (and optionally the adversary) at the
           start of the episode.
           - Adds the selected actions to the agent's replay memory for future training.
           - Adjusts the agent's exploration-exploitation balance (epsilon) based on the episode's performance.
           - Periodically updates the target network.

       Returns:
           None
       """
    print(f"Started episode {episode} out of {num_episodes}")
    env.reset()  # Reset the environment for a new episode
    taken_actions = set()  # Reset the made actions
    previous_action = None
    state = preprocess_image(load_image(env.get_screenshot()))  # Get and preprocess the initial state
    move_count = 0  # Track the number of moves in the current episode
    players = [(agent, "Agent"), (adversary, "Adversary") if adversary else None]

    for player, role in itertools.cycle(players):
        if player is None:  # Skip if there's no adversary
            continue

        if adversary:  # If there is an adversary, log whose move it is
            print(f"{role}'s move")
        result = _make_move(player, env, state, taken_actions, batch_size,
                            previous_action if role == "Adversary" else None)
        if result is None:
            break

        state, previous_action = result

    # Adjust exploration if the tower fell too quickly
    update_epsilon(agent, efficiency_threshold, move_count)
    # Update the target network periodically
    if episode % target_update == 0:
        agent.update_target_net()


def _make_move(agent, env, state, taken_actions, batch_size, previous_action=None):
    if previous_action is None:
        action = agent.select_action(state, taken_actions)  # Agent's action
    else:
        action = agent.select_action(state, taken_actions, previous_action)  # Adversary's action

    if action is None:
        print("No action to take. Ending the episode")
        return

    next_state, is_fallen = env.step(action)
    next_state = preprocess_image(load_image(next_state))

    if previous_action is None:
        agent.memory.push(state, action, calculate_reward(action, is_fallen), next_state, is_fallen)
        agent.optimize_model(batch_size)

    if is_fallen:  # Stop the episode if the tower has fallen
        print("The tower has fallen. Ending the episode")
        return

    return next_state, action


if __name__ == "__main__":
    # First phase: the agent trains against itself, starting from scratch
    # training_loop(if_load_weights=False)

    # Second phase: the agent trains against the random-strategy adversary
    training_loop(if_training_against_adversary=True, strategy=RandomStrategy())

    # Third phase: the agent trains against a pessimistic-strategy adversary
    training_loop(if_training_against_adversary=True, strategy=PessimisticStrategy())

    # Fourth phase: the agent trains against an optimistic-strategy adversary
    training_loop(if_training_against_adversary=True, strategy=OptimisticStrategy())
