import itertools

from matplotlib import pyplot as plt

import utils
from adversary.adversary import Adversary
from adversary.strategy import RandomStrategy, OptimisticStrategy, PessimisticStrategy
from environment.environment import Environment
from monte_carlo_tree_search.mcts_agent import MCTSAgent


def plot_winrate(agent, strategies, num_games_intervals):
    """
    Plays games with the agent against different strategies and plots the win rate as a function of the number of games.

    Args:
        agent (MCTSAgent): The MCTS agent to play against.
        strategies (list): A list of strategies to test against.
        num_games_intervals (list): A list specifying the number of games to play for each interval.

    Returns:
        None
    """
    win_rates = {strategy.__class__.__name__: [] for strategy in strategies}

    for num_games in num_games_intervals:
        for strategy in strategies:
            strategy_name = strategy.__class__.__name__
            print(f"Playing {num_games} games against {strategy_name}...")
            win_rate = evaluate_winrate(agent, strategy, num_games)
            win_rates[strategy_name].append(win_rate)
            print(f"Win rate against {strategy_name} after {num_games} games: {win_rate:.2f}")

    # Recalculating the number of games for the plot
    num_games = []
    num_game = 0
    for i in range(len(num_games_intervals)):
        if i == 0:
            num_game = num_games_intervals[i]
        else:
            num_game += num_games_intervals[i]
        num_games.append(num_game)

    # Plotting the win rates
    plt.figure(figsize=(12, 8))
    for strategy_name, rates in win_rates.items():
        plt.plot(num_games, rates, label=f'Against {strategy_name}')
    plt.xlabel('Number of Games Played')
    plt.ylabel('Win Rate')
    plt.title('Win Rate vs Number of Games Played')
    plt.legend()
    plt.grid(True)
    plt.show()


def evaluate_winrate(agent, strategy, num_games):
    """
    Evaluates the win rate of the MCTS agent against a specific strategy.

    Args:
        agent (MCTSAgent): The agent to be tested.
        strategy (Strategy): The adversary strategy to evaluate against.
        num_games (int): Number of test games.

    Returns:
        float: The win rate of the agent against the strategy.
    """
    wins = 0
    adversary = Adversary(strategy=strategy)
    env = Environment()
    initial_state = utils.get_state_from_image(env.get_screenshot())

    for i in range(1, num_games + 1):
        print(f"Starting game {i}")
        env.reset()
        taken_actions = set()  # Reset the actions taken
        state = initial_state

        for _ in itertools.count():
            # MCTS Agent's turn
            agent_action = agent.select_action(state, taken_actions)
            if agent_action is None:
                break
            print(f"Agent chose action {agent_action}")

            screenshot_filename, is_fallen = env.step((agent_action[0], utils.INT_TO_COLOR[agent_action[1]]))
            if is_fallen:
                wins += 1
                break
            state = utils.get_state_from_image(screenshot_filename)
            taken_actions.add(agent_action)

            # Adversary's turn
            adversary_action = adversary.select_action(state, taken_actions, agent_action)
            if adversary_action is None:
                wins += 1
                break
            print(f"Adversary chose action {adversary_action}")

            screenshot_filename, is_fallen = env.step((adversary_action[0], utils.INT_TO_COLOR[adversary_action[1]]))
            if is_fallen:
                break
            state = utils.get_state_from_image(screenshot_filename)
            taken_actions.add(adversary_action)

    win_rate = wins / num_games
    return win_rate


def plot_mcts_winrate():
    """
    Plays games with the MCTS agent against different strategies and plots the win rate.
    """
    env = Environment()
    mcts_agent = MCTSAgent(env)

    strategies = [RandomStrategy(), OptimisticStrategy(), PessimisticStrategy()]
    num_games_intervals = [10, 15, 20, 30]  # Number of games to measure the win rate for

    plot_winrate(mcts_agent, strategies, num_games_intervals)


if __name__ == "__main__":
    plot_mcts_winrate()
