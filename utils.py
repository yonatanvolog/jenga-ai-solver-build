import torchvision.transforms as transforms
from PIL import Image

from environment.environment import MAX_BLOCKS_IN_LEVEL, MAX_LEVEL, INT_TO_COLOR, SCREENSHOT_SHAPE


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
        transforms.Resize(SCREENSHOT_SHAPE),
        transforms.ToTensor(),        # Convert to tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])
    return preprocess(image).unsqueeze(0)  # Add batch dimension


def get_state_from_image(filename):
    """
    Loads the image in the filename and preprocesses the input image by converting it to grayscale, resizing,
    converting to a tensor, and normalizing it.

    Args:
        filename (str): The path to the image file.

    Returns:
        torch.Tensor: The preprocessed image as a tensor with a batch dimension. This image represents the state
                      of Jenga
    """
    return preprocess_image(load_image(filename))


def get_possible_actions(taken_actions=set()):
    """
    Returns a list of possible actions that can be taken in the Jenga game, excluding those that have already been
    taken.

    Each action is represented as a tuple containing the level and color of the block to be removed.

    Args:
        taken_actions (set, optional): A set of actions that have already been taken. Each action in the set is a tuple
                                       of the form (level, color), where `level` is an integer representing the level
                                       of the block in the Jenga tower, and `color` is an integer representing the
                                       color of the block. Defaults to an empty set, meaning no actions have been taken.

    Returns:
        list: A list of possible actions that can be taken, excluding those in `taken_actions`. Each action is a tuple
              of the form (level, color).
    """
    return list({(level, color) for level in range(MAX_LEVEL) for color in range(MAX_BLOCKS_IN_LEVEL)} - taken_actions)


def format_action(action):
    """
    Formats the given action by converting the block's color from its integer representation
    to its corresponding string representation.

    Args:
        action (tuple): A tuple representing the action to take in the Jenga game. The action is a
                        tuple of the form (level, color), where `level` is an integer representing
                        the level of the block in the Jenga tower, and `color` is an integer
                        representing the color of the block (0 for yellow, 1 for blue, 2 for green).

    Returns:
        tuple: A formatted tuple representing the action in the form (level, color), where `level`
               is the same integer as the input, and `color` is the string representation of the block color
               ('y', 'b', or 'g').
    """
    return action[0], INT_TO_COLOR[action[1]]


def calculate_reward(action, previous_stability, current_stability):
    """
    Calculates the reward for the agent's action with a small bonus for minor instability and no penalty in such cases.

    Args:
        action (tuple): The action taken by the agent, including the level and color.
        previous_stability (float): Stability before the move.
        current_stability (float): Stability after the move.

    Returns:
        float: The calculated reward.
    """
    level, color = action

    # Base reward based on the level of the block removed
    base_reward = level

    # Calculate the stability difference
    stability_diff = previous_stability - current_stability if previous_stability else -current_stability
    print(stability_diff)

    # Give a small bonus if stability changed only slightly (e.g., within a threshold)
    stability_bonus = 5 if -3 <= stability_diff <= 1 else 0

    # No stability penalty if the reduction is minor, or no bonus if the stability is improved
    stability_penalty = 0 if stability_bonus or stability_diff > 0 else stability_diff * 10

    # Combine the rewards and penalties
    reward = max(base_reward + stability_bonus + stability_penalty, -20)

    return reward
