import itertools
import time
from enum import Enum

import utils
from adversary.adversary import Adversary
from adversary.strategy import RandomStrategy
from hierarchical_deep_q_learning.hierarchical_deep_q_agent import HierarchicalDQNAgent
from dev_environment.environment import Environment, COLOR_TO_INT
from hierarchical_sarsa_deep_q_learning.hierarchical_sarsa_agent import HierarchicalSARSAAgent


class PlayerType(Enum):
    """
    Enum representing different types of players in the game.
    """
    RANDOM = 0
    DQN = 1
    SARSA = 2
    HUMAN = 3


def player_factory(player_type):
    """
    Factory function to create different types of players based on PlayerType.

    Args:
        player_type (PlayerType): The type of player to create.

    Returns:
        Adversary, HierarchicalDQNAgent, or HierarchicalSARSAAgent:
        A player of the specified type.
    """
    if player_type is PlayerType.RANDOM:
        return Adversary(strategy=RandomStrategy())
    elif player_type is PlayerType.DQN:
        # randomizing the actions a bit to make the agents use other actions
        agent = HierarchicalDQNAgent(epsilon_start=0.3)
        agent.load_model(level_1_path="hierarchical_deep_q_learning/weights/level_1_game.pth",
                         level_2_path="hierarchical_deep_q_learning/weights/level_2_game.pth")
        return agent
    elif player_type is PlayerType.SARSA:
        agent = HierarchicalSARSAAgent(epsilon_start=0.3)
        agent.load_model(level_1_path="hierarchical_sarsa_deep_q_learning/weights/level_1_game.pth",
                         level_2_path="hierarchical_sarsa_deep_q_learning/weights/level_2_game.pth")
        return agent
    return


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
    return player.select_action(state, taken_actions)


def _make_ai_move(player_type, player, state, taken_actions, previous_action, env):
    """
    Facilitates making a move for a specified player type, executing the action in the game environment, and updating
    the state.

    Args:
        player_type (PlayerType): The type of player executing the move (e.g., AI agent or human).
        player (object): The player instance making the move, either an AI agent or a human agent.
        state (torch.Tensor): The current state of the game (e.g., a visual representation of the environment).
        taken_actions (set): A set of already performed actions, ensuring no duplicate actions.
        previous_action (tuple): The last action taken in the game, used to track progress.
        env (Environment): The game environment where actions are executed and game state is updated.
    """

    action = select_action(player_type, player, state, taken_actions, previous_action)

    if action is None:
        return  # End the game if no action can be taken

    env.step(utils.format_action(action), if_get_state=False)


def play(env, player_1_type, player_2_type, num_games):
    """
    Simulates a series of games between two players.

    Args:
        env (Environment): The Jenga environment.
        player_1_type (PlayerType): The type of the first player.
        player_2_type (PlayerType): The type of the second player.
        num_games (int): The number of games to simulate.
    """
    player_1 = player_factory(player_1_type)
    player_2 = player_factory(player_2_type)

    initial_state = utils.get_state_from_image(env.get_screenshot())

    # Loop through the specified number of games
    for i in range(1, num_games + 1):
        taken_actions = set()  # Track the actions taken
        state = initial_state  # Initialize the state for the game
        previous_action = None

        # Loop through the moves made in the game until completion
        for player_type, player, player_index in itertools.cycle([(player_1_type, player_1, 0),
                                                                  (player_2_type, player_2, 1)]):
            # Send command indicating it's the current player's turn
            print(f"Sending player_turn for Player {player_index}")
            env.send_command(
                f"player_turn {player_type.value} {player_index} {i}")

            if player_type != PlayerType.HUMAN:
                _make_ai_move(player_type, player, state, taken_actions, previous_action, env)

            # Wait for the "finished_move" command from Unity
            command = env.listen_for_commands()
            print(f"Received command: {command}")
            if command.startswith("finished_move"):

                # Check if the tower has fallen
                time.sleep(0.5)
                is_fallen = env.is_fallen()
                # Retrieve the screenshot after performing the action
                time.sleep(0.5)
                state = utils.get_state_from_image(env.get_screenshot())
                _, level, color = command.split()
                previous_action = (level, COLOR_TO_INT[color])
                taken_actions.add(previous_action)

                if is_fallen:
                    print(f"Player {player_index} lost the game!")
                    time.sleep(2) # Give players time to see who has won
                    env.reset()
                    break

            if command.startswith("end_game"):
                env.reset()
                env.toggle_menu()
                return

        # Show menu only when the last round ended
        if i == num_games:
            env.toggle_menu()


def listen_for_start():
    """
    Main game loop to start the game and determine player types and number of rounds.
    """
    env = Environment(relative_path_to_screenshots="./screenshots",
                      unity_exe_path="./prod_environment/jenga-game.exe")
    env.reset()
    env.toggle_menu()

    while True:
        start_command = env.listen_for_commands()
        if start_command.startswith("start"):
            _, p1_type, p2_type, num_games = start_command.split()
            num_games = int(num_games)
            p1_type = PlayerType(int(p1_type))
            p2_type = PlayerType(int(p2_type))
            play(env, p1_type, p2_type, num_games)


if __name__ == "__main__":
    listen_for_start()
