import itertools
import torchvision.transforms as transforms
from PIL import Image
from deep_q_learning.deep_q_agent import HierarchicalDQNAgent
from environment.environment import Environment

# Mapping from color to integer for Jenga blocks
COLOR_TO_INT = {"y": 0, "b": 1, "g": 2}


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


def training_loop():
    """
    Runs the training loop for the HierarchicalDQNAgent in a Jenga environment.

    The agent interacts with the environment over a series of episodes, learning
    to select the best blocks to remove. The training loop includes steps for
    action selection, environment interaction, state preprocessing, experience
    storage, and model optimization. The target network is updated periodically.
    """
    num_episodes = 100  # Number of episodes to train the agent
    batch_size = 10     # Number of experiences to sample for each training step
    target_update = 10  # Frequency (in episodes) to update the target network

    # Initialize the agent and environment
    agent = HierarchicalDQNAgent(input_shape=(128, 64), num_actions_level_1=10, num_actions_level_2=3)
    env = Environment()
    env.set_timescale(100)  # Speed up the simulation

    for episode in range(num_episodes):
        print("Started a new episode")
        env.reset()  # Reset the environment for a new episode
        state = preprocess_image(load_image(env.get_screenshot()))  # Get and preprocess the initial state

        for _ in itertools.count():
            action = agent.select_action(state)  # Select an action based on the current state
            next_state, is_fallen = env.step(action)  # Perform the action in the environment
            next_state = preprocess_image(load_image(next_state))  # Preprocess the new state

            # Store the experience in replay memory
            agent.memory.push(state, (action[0], COLOR_TO_INT[action[1]]), action[0], next_state, is_fallen)
            state = next_state  # Update the state

            # Optimize the model based on the collected experiences
            agent.optimize_model(batch_size)

            if is_fallen:  # Stop the episode if the tower has fallen
                print("Stopped this episode")
                break

        if episode % target_update == 0:  # Update the target network periodically
            agent.update_target_net()


if __name__ == "__main__":
    training_loop()
