import itertools
import utils
from adversary.adversary import Adversary
from adversary.strategy import RandomStrategy, PessimisticStrategy, OptimisticStrategy
from environment.environment import Environment
from hierarchical_sarsa_deep_q_learning.hierarchical_sarsa_agent import HierarchicalSARSAAgent


def update_epsilon(agent, efficiency_threshold, move_count):
    """
    Adjusts the epsilon value for the agent's epsilon-greedy policy based on the efficiency of the current strategy.

    This function modifies the agent's epsilon, which controls the balance between exploration and exploitation.
    If the number of moves made before the tower falls is below the efficiency threshold, epsilon is increased
    to promote more exploration in the next episode. Conversely, if the move count meets or exceeds the threshold,
    epsilon is decreased to encourage exploitation of the current strategy.

    Args:
        agent (HierarchicalSARSAAgent): The agent whose epsilon value is being adjusted.
        efficiency_threshold (int): The minimum number of moves before the tower falls to consider the strategy
                                    efficient.
        move_count (int): The number of moves made in the current episode before the tower fell.
    """
    if move_count < efficiency_threshold:
        agent.epsilon = min(1.0, agent.epsilon * 1.1)  # Increase epsilon to promote exploration
        print(f"Increased exploration: epsilon = {agent.epsilon}")
    else:
        agent.epsilon = max(agent.epsilon_end, agent.epsilon * 0.9)  # Decrease epsilon to promote exploitation
        print(f"Decreased exploration: epsilon = {agent.epsilon}")


def training_loop(agent=None, env=None, num_episodes=50, batch_size=10, efficiency_threshold=10,
                  if_load_weights=True, level_1_path="weights/level_1.pth", level_2_path="weights/level_2.pth",
                  if_training_against_adversary=False, strategy=RandomStrategy()):
    """
    Runs the training loop for the HierarchicalSARSAAgent using SARSA in a Jenga environment.

    The agent interacts with the environment over a series of episodes. If `adversary_training` is True,
    the agent will train against an adversary initialized with the agent's weights from the last phase.

    Args:
        agent (HierarchicalSARSAAgent): The agent.
        env (Environment): The Jenga environment.
        num_episodes (int): Number of episodes to run.
        batch_size (int): Batch size.
        if_load_weights (bool): Whether to load pre-existing model weights if they exist or start from scratch.
        level_1_path (str): Path to the weights of the first DQN.
        level_2_path (str): Path to the weights of the second DQN.
        if_training_against_adversary (bool): Whether to train against an adversary.
        strategy (Strategy): Strategy for the adversary to take.
        efficiency_threshold (int): The minimum number of moves before the tower falls to consider the strategy efficient.
    """
    print("Starting a new SARSA training loop")

    # Initialize the agent and environment
    if agent is None:
        agent = HierarchicalSARSAAgent()
    if env is None:
        env = Environment()

    # Load model weights if they exist
    if if_load_weights:
        try:
            agent.load_model(level_1_path, level_2_path)
        except FileNotFoundError:
            print("No previous model found. Starting from scratch")

    # Create an adversary based on the agent's current model
    adversary = None
    if if_training_against_adversary:
        adversary = Adversary(strategy)
    if adversary:
        print(f"The agent is training against an adversary with {adversary.__class__.__name__} strategy")

    for episode in range(1, num_episodes + 1):
        _run_episode(adversary, agent, batch_size, efficiency_threshold, env, episode, num_episodes)

    # Save model weights at the end of the training session
    agent.save_model(level_1_path, level_2_path)


def _run_episode(adversary, agent, batch_size, efficiency_threshold, env, episode, num_episodes):
    """
    Runs a single episode of training for the HierarchicalSARSAAgent in the Jenga environment.

    During the episode, the agent (and optionally an adversary) interacts with the environment, selecting actions
    and optimizing its model based on the results of those actions. The episode continues until the Jenga tower
    falls, or no more valid actions are available.

    Args:
        adversary (Adversary or None): An optional adversary agent that may take turns with the main agent.
        agent (HierarchicalSARSAAgent): The main agent being trained.
        batch_size (int): The number of experiences to sample from replay memory for training.
        efficiency_threshold (int): The minimum number of moves before the tower falls to consider the strategy
                                    efficient.
        env (Environment): The environment representing the Jenga game.
        episode (int): The current episode number.
        num_episodes (int): The total number of episodes in the training session.
    """
    print(f"Started episode {episode} out of {num_episodes}")
    env.reset()  # Reset the environment for a new episode
    taken_actions = set()  # Reset the made actions
    previous_action = None
    previous_stability = None
    state = utils.get_state_from_image(env.get_screenshot())  # Get and preprocess the initial state
    move_count = 0  # Track the number of moves in the current episode
    players = [(agent, "Agent"), (adversary, "Adversary")]

    for player, role in itertools.cycle(players):
        if player is None:  # Skip if there's no adversary
            continue

        if adversary:  # If there is an adversary, log whose move it is
            print(f"{role}'s move")
        result = _make_move(player, env, state, taken_actions, batch_size,
                            previous_action if role == "Adversary" else None,
                            previous_stability)
        if result is None:
            break

        state, previous_action, previous_stability = result

    # Adjust exploration if the tower fell too quickly
    update_epsilon(agent, efficiency_threshold, move_count)


def _make_move(agent, env, state, taken_actions, batch_size, previous_action=None, previous_stability=None):
    """
    Executes a move in the Jenga game by either the agent or the adversary using SARSA.

    This function selects an action based on the agent's or adversary's strategy, interacts with the environment
    by performing the action, and updates the state accordingly. The function also updates the agent's replay memory
    using SARSA transitions and optimizes its model.

    Args:
        agent (HierarchicalSARSAAgent or Adversary): The agent or adversary making the move.
        env (Environment): The environment representing the Jenga game.
        state (torch.Tensor): The current state of the environment.
        taken_actions (set): A set of actions that have already been taken, to avoid repetition.
        batch_size (int): The number of experiences to sample from replay memory for training.
        previous_action (tuple or None): The previous action taken by the agent. If None, the move is made by the agent;
                                         otherwise, it's made by the adversary and uses the `previous_action` to inform
                                         the selection of the next action.

    Returns:
        tuple or None: If the move is successful and the game continues, returns a tuple containing the next state and
                       the action taken (next_state, action). If no action can be taken or the tower falls, returns None
                       to indicate that the episode should end.
    """
    if previous_action is None:
        # SARSA: Select current action for the agent
        action = agent.select_action(state, taken_actions)
    else:
        # Adversary's action based on SARSA
        action = agent.select_action(state, taken_actions, previous_action)

    if action is None:
        print("No action to take. Ending the episode")
        return

    # Record the current stability before taking the action
    current_stability = env.get_average_max_tilt_angle()

    # Perform the selected action
    screenshot_filename, is_fallen = env.step(utils.format_action(action))
    next_state = utils.get_state_from_image(screenshot_filename)

    # If the tower has not fallen, select the next action (SARSA)
    print("Selecting the next action and not actually making it for now")
    if previous_action is None:
        next_action = agent.select_action(next_state, taken_actions)
    else:
        next_action = agent.select_action(next_state, taken_actions, previous_action)

    if previous_action is None:
        # Calculate the reward and store the SARSA transition
        reward = utils.calculate_reward(action, previous_stability, current_stability)
        agent.memory.sarsa_push(state, action, reward, next_state, next_action, is_fallen)
        agent.optimize_model(batch_size)

    if is_fallen:
        # Stop the episode if the tower has fallen
        print("The tower has fallen. Ending the episode")
        return

    # Return the next state, the current action, and the updated stability
    return next_state, next_action, current_stability


if __name__ == "__main__":
    # Here the agent learns to play against humans and the strategies, to use the resulting weights afterwards

    # First phase: the agent trains against an optimistic-strategy adversary
    training_loop(if_load_weights=False, if_training_against_adversary=True, strategy=OptimisticStrategy())

    # Second phase: the agent trains against a pessimistic-strategy adversary
    training_loop(if_training_against_adversary=True, strategy=PessimisticStrategy())

    # Third phase: the agent trains against the random-strategy adversary
    training_loop(if_training_against_adversary=True, strategy=RandomStrategy())

    # Last phase: the agent trains against itself
    training_loop(num_episodes=10)
    training_loop(num_episodes=15)
    training_loop(num_episodes=20)
