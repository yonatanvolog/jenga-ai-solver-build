import itertools
import socket
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
    RANDOM = 0
    DQN = 1
    SARSA = 2
    GSBAS = 3
    HUMAN = 4


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
        agent.load_model(level_1_path="hierarchical_deep_q_learning/level_1.pth",
                         level_2_path="hierarchical_deep_q_learning/level_2.pth")
        return agent
    elif player_type is PlayerType.SARSA:
        agent = HierarchicalSARSAAgent()
        agent.load_model(level_1_path="hierarchical_sarsa_deep_q_learning/level_1.pth",
                         level_2_path="hierarchical_sarsa_deep_q_learning/level_2.pth")
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


def _make_move(player_type, player, state, taken_actions, previous_action, env):
    """
    Facilitates making a move for a specified player type, executing the action in the game environment, and updating the state.

    Args:
        player_type (PlayerType): The type of player executing the move (e.g., AI agent or human).
        player (object): The player instance making the move, either an AI agent or a human agent.
        state (torch.Tensor): The current state of the game (e.g., a visual representation of the environment).
        taken_actions (set): A set of already performed actions, ensuring no duplicate actions.
        previous_action (tuple): The last action taken in the game, used to track progress.
        env (Environment): The game environment where actions are executed and game state is updated.

    Returns:
        Optional[tuple]:
            - next_state (torch.Tensor): The updated state of the game after performing the action.
            - action (tuple): The action taken in the current move, represented as (level, color).
        If no action can be taken or the tower falls, the function returns None.
    """

    action = select_action(player_type, player, state, taken_actions, previous_action)

    if action is None:
        return  # End the game if no action can be taken

    # Take the action and get the updated state
    screenshot_filename, is_fallen = env.step(utils.format_action(action))

    if is_fallen:
        return  # End the game if the tower has fallen

    # Update the game state and record the action taken
    next_state = utils.get_state_from_image(screenshot_filename)
    taken_actions.add(action)

    return next_state, action


def play(env, player_1_type, player_2_type, num_games):
    """
    Simulates a series of games between two players.

    Args:
        env (Environment): The Jenga environment.
        player_1_type (PlayerType): The type of the first player.
        player_2_type (PlayerType): The type of the second player.
        num_games (int): The number of games to simulate.
    """
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
        for player_type, player in itertools.cycle([(player_1_type, player_1), (player_2_type, player_2)]):
            result = _make_move(player_type, player, state, taken_actions, previous_action, env)
            if result is None:
                break
            state, previous_action = result


def start_listener():
    """
    Listens for commands via TCP using the environment function. When receiving "start <player1_type> <player2_type> <num_games>",
    it starts the play function and stops listening until the game finishes.
    """
    env = Environment(unity_exe_path=None, relative_path_to_screenshots="environment/screenshots")

    while True:
        command = env.listen_for_commands()
        print(command)
        if not command.startswith("start"):
            continue
        # Parse the command
        parts = command.split()
        if len(parts) != 4:
            continue
        player_1_type = PlayerType(int(parts[1]))
        player_2_type = PlayerType(int(parts[2]))
        num_games = int(parts[3])

        # Start the game
        play(env, player_1_type, player_2_type, num_games)


if __name__ == "__main__":
    start_listener()
