import itertools
from enum import Enum

import utils
from adversary.adversary import Adversary
from adversary.strategy import RandomStrategy
from hierarchical_deep_q_learning.hierarchical_deep_q_agent import HierarchicalDQNAgent
from environment.environment import Environment
from hierarchical_sarsa_deep_q_learning.hierarchical_sarsa_agent import HierarchicalSARSAAgent
from greedy_simulation_based_action_search.gsbas_agent import GSBASAgent
from human_agent import HumanAgent


class PlayerType(Enum):
    """
    Enum representing different types of players in the game.

    Attributes:
        RANDOM: Represents a player that follows a random strategy.
        DQN: Represents a player that uses a Hierarchical Deep Q-Network agent.
        SARSA: Represents a player that uses a Hierarchical SARSA agent.
        GSBAS: Represents a player that uses the Greedy Simulation-Based Action Search agent.
        HUMAN: Represents a human player.
    """
    RANDOM = "Random"
    DQN = "DQN"
    SARSA = "SARSA"
    GSBAS = "GSBAS"
    HUMAN = "Human"


def player_factory(player_type, env):
    """
    Factory function to create different types of players based on PlayerType.

    Args:
        player_type (PlayerType): The type of player to create.
        env (Environment): The game environment that agents or human players interact with.

    Returns:
        Adversary, HierarchicalDQNAgent, HierarchicalSARSAAgent, GSBASAgent, or HumanAgent:
        A player of the specified type.
    """
    if player_type is PlayerType.RANDOM:
        return Adversary(strategy=RandomStrategy())
    elif player_type is PlayerType.DQN:
        agent = HierarchicalDQNAgent()
        agent.load_model(level_1_path="../hierarchical_deep_q_learning/level_1.pth",
                         level_2_path="../hierarchical_deep_q_learning/level_2.pth")
        return agent
    elif player_type is PlayerType.SARSA:
        agent = HierarchicalSARSAAgent()
        agent.load_model(level_1_path="../hierarchical_sarsa_deep_q_learning/level_1.pth",
                         level_2_path="../hierarchical_sarsa_deep_q_learning/level_2.pth")
        return agent
    elif player_type is PlayerType.GSBAS:
        return GSBASAgent(env)
    return HumanAgent(env)


def select_action(player_type, player, state, taken_actions, previous_action):
    """
    Selects an action for the player based on the player type.

    Args:
        player_type (PlayerType): The type of player making the move.
        player (Adversary, HierarchicalDQNAgent, HierarchicalSARSAAgent, GSBASAgent, HumanAgent): The player instance.
        state (torch.Tensor): The current state of the environment.
        taken_actions (Set[Tuple[int, int]]): Already performed actions.
        previous_action (tuple): The previous action taken, used for adversaries.

    Returns:
        tuple: The selected action (level, color).
    """
    if player_type is PlayerType.RANDOM:
        return player.select_action(state, taken_actions, previous_action)
    elif player_type in [PlayerType.DQN, PlayerType.SARSA]:
        return player.select_action(state, taken_actions, if_allow_exploration=False)
    else:
        return player.select_action(state, taken_actions)


def play(player_1_type, player_2_type, num_games):
    """
    Simulates a series of games between two players.

    Args:
        player_1_type (PlayerType): The type of the first player.
        player_2_type (PlayerType): The type of the second player.
        num_games (int): The number of games to simulate.
    """
    env = Environment()
    player_1 = player_factory(player_1_type, env)
    player_2 = player_factory(player_2_type, env)

    env.reset()  # Reset the environment
    initial_state = utils.get_state_from_image(env.get_screenshot())

    # Loop through the specified number of games
    for _ in range(num_games):
        env.reset()  # Reset the environment for each game
        taken_actions = set()  # Track the actions taken
        state = initial_state  # Initialize the state for the game
        previous_action = None

        # Loop through the moves made in the game until completion
        for _ in itertools.count():
            # Iterate between player 1 and player 2
            for player_type, player in [(player_1_type, player_1), (player_2_type, player_2)]:
                previous_action = select_action(player_type, player, state, taken_actions, previous_action)

                if previous_action is None:
                    break  # End the game if no action can be taken

                # Take the action and get the updated state
                screenshot_filename, is_fallen = env.step(utils.format_action(previous_action))

                if is_fallen:
                    break  # End the game if the tower has fallen

                # Update the game state and record the action taken
                state = utils.get_state_from_image(screenshot_filename)
                taken_actions.add(previous_action)
