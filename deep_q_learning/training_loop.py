import itertools

import torchvision.transforms as transforms
from PIL import Image

from deep_q_learning.deep_q_agent import HierarchicalDQNAgent
from environment.environment import Environment


def load_image(filename):
    # Open the image file
    image = Image.open(filename)
    return image


def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((128, 64)),  # Resize to 128x64 pixels
        transforms.ToTensor(),        # Convert to tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])
    return preprocess(image).unsqueeze(0)  # Add batch dimension


def training_loop():
    num_episodes = 1000
    batch_size = 32
    target_update = 10

    agent = HierarchicalDQNAgent(input_shape=(128, 64), num_actions_level_1=10, num_actions_level_2=3)
    env = Environment()

    for episode in range(num_episodes):
        state = preprocess_image(load_image(env.reset()))

        for _ in itertools.count():
            action = agent.select_action(state)
            reward = action[0]  # num_level
            next_state, is_fallen = env.step(action)
            next_state = preprocess_image(load_image(next_state))

            agent.memory.push(state, action, reward, next_state, is_fallen)
            state = next_state

            agent.optimize_model(batch_size)

            if is_fallen:
                break

        if episode % target_update == 0:
            agent.update_target_net()


if __name__ == "__main__":
    training_loop()
