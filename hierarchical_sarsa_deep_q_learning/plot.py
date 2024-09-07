import itertools
import random
import matplotlib.pyplot as plt
import utils
from adversary.adversary import Adversary
from adversary.strategy import RandomStrategy, PessimisticStrategy, OptimisticStrategy
from hierarchical_sarsa_agent import HierarchicalSARSAAgent  # Import SARSA Agent
from environment.environment import Environment
from training_loop import training_loop


def train_and_plot_winrate(agent, env, strategies, episode_intervals, num_tests=20, batch_size=10, target_update=10,
                           if_training_against_adversary=False):
    """
    Trains the SARSA agent for increasingly longer numbers of episodes and plots the win rate against each strategy.

    Args:
        agent (HierarchicalSARSAAgent): The SARSA agent to be trained.
        env (Environment): Jenga environment.
        strategies (list): A list of strategies to train and evaluate against.
        episode_intervals (list): A list of episode counts to train the agent on.
        num_tests (int): Number of test episodes to evaluate win rates after training.
        batch_size (int): Batch size for training.
        target_update (int): Number of episodes after which to update the target network.
        if_training_against_adversary (bool): Specifies if the agent is training against itself (False) or a strategy
                                              (True).
    """
    win_rates = {strategy.__class__.__name__: [] for strategy in strategies}

    for i in range(len(episode_intervals)):
        # Train the agent using SARSA
        print(f"Training for {episode_intervals[i]} episodes...")
        training_loop(
            agent=agent,
            env=env,
            num_episodes=episode_intervals[i],
            batch_size=batch_size,
            if_load_weights=False if i == 0 else True,
            level_1_path="level_1_plots.pth",
            level_2_path="level_2_plots.pth",
            if_training_against_adversary=if_training_against_adversary
        )

        for strategy in strategies:
            strategy_name = strategy.__class__.__name__
            print(f"Evaluating win rate against {strategy_name}...")
            win_rate = evaluate_winrate(agent, env, strategy, num_tests)
            win_rates[strategy_name].append(win_rate)
            print(f"Win rate against {strategy_name}: {win_rate:.2f}")

    # Recalculating the episodes for the plot
    episodes = []
    num_episode = 0
    for i in range(len(episode_intervals)):
        if i == 0:
            num_episode = episode_intervals[i]
        else:
            num_episode += episode_intervals[i]
        episodes.append(num_episode)

    # Plotting the win rates
    plt.figure(figsize=(12, 8))
    for strategy_name, rates in win_rates.items():
        plt.plot(episodes, rates, label=f'Against {strategy_name}')

    plt.xlabel('Number of Training Episodes')
    plt.ylabel('Win Rate')
    plt.title(f"Win Rate After {num_tests} Games Against Strategies as Function of Number Training Episodes Against "
              f"{'Itself' if not if_training_against_adversary else 'Random Strategy'}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"sarsa_winrate_as_function_of_num_training_episodes_against_"
                f"{'itself' if not if_training_against_adversary else 'random'}.png")


def evaluate_winrate(agent, env, strategy, num_tests):
    """
    Evaluates the win rate of the SARSA agent against a specific strategy.

    Args:
        agent (HierarchicalSARSAAgent): The SARSA agent to be tested.
        env (Environment): Jenga environment.
        strategy (Strategy): The adversary strategy to evaluate against.
        num_tests (int): Number of test episodes.

    Returns:
        float: The win rate of the agent against the strategy.
    """
    wins = 0
    adversary = Adversary(strategy=strategy)
    initial_state = utils.get_state_from_image(env.get_screenshot())

    for i in range(1, num_tests + 1):
        print(f"Starting game {i}")
        env.reset()
        taken_actions = set()  # Reset the actions taken
        state = initial_state
        previous_action = None
        if_first_player_is_agent = random.choice([True, False])  # Randomly choosing who moves first
        if if_first_player_is_agent:
            first_player = agent
            second_player = adversary
        else:
            first_player = adversary
            second_player = agent

        for _ in itertools.count():
            if if_first_player_is_agent:
                previous_action = first_player.select_action(state, taken_actions, if_allow_exploration=False)
            else:
                previous_action = first_player.select_action(state, taken_actions, previous_action)

            if previous_action is None:
                print("No actions to take. Stopping this game")
                break

            screenshot_filename, is_fallen = env.step(utils.format_action(previous_action))
            if is_fallen:
                if not if_first_player_is_agent:
                    wins += 1
                print("The tower has fallen. Stopping this game")
                break
            state = utils.get_state_from_image(screenshot_filename)
            taken_actions.add(previous_action)

            if if_first_player_is_agent:
                previous_action = second_player.select_action(state, taken_actions, previous_action)
            else:
                previous_action = second_player.select_action(state, taken_actions, if_allow_exploration=False)

            if previous_action is None:
                print("No actions to take. Stopping this game")
                break

            screenshot_filename, is_fallen = env.step(utils.format_action(previous_action))
            if is_fallen:
                if if_first_player_is_agent:
                    wins += 1
                print("The tower has fallen. Stopping this game")
                break
            state = utils.get_state_from_image(screenshot_filename)
            taken_actions.add(previous_action)

    win_rate = wins / num_tests
    return win_rate


def plot_1(env):
    """
    Trains the SARSA agent against itself and plots the win rate against the strategies.
    """
    agent = HierarchicalSARSAAgent(input_shape=(128, 64), num_actions_level_1=12, num_actions_level_2=3)
    strategies = [RandomStrategy(), OptimisticStrategy(), PessimisticStrategy()]
    episode_intervals = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    train_and_plot_winrate(agent, env, strategies, episode_intervals)


def plot_2(env):
    """
    Trains the SARSA agent against RandomStrategy and plots the win rate against the strategies.
    """
    agent = HierarchicalSARSAAgent(input_shape=(128, 64), num_actions_level_1=12, num_actions_level_2=3)
    strategies = [RandomStrategy(), OptimisticStrategy(), PessimisticStrategy()]
    episode_intervals = [10, 10, 10, 10, 10, 10]
    train_and_plot_winrate(agent, env, strategies, episode_intervals, if_training_against_adversary=True)


if __name__ == "__main__":
    env = Environment()
    plot_1(env)
    plot_2(env)
