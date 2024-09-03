import itertools
from environment.environment import Environment
from deep_q_learning.adversary import Adversary
from deep_q_learning.strategy import RandomStrategy, PessimisticStrategy, OptimisticStrategy
from monte_carlo_tree_search.mcts_agent import MCTSAgent
from utils import get_state_from_image


def mcts_training_loop(num_episodes=50, agent_simulations=1000, exploration_weight=1.0, if_training_against_adversary=False,
                       strategy=RandomStrategy(), efficiency_threshold=10):
    """
    Runs the training loop for the MCTSAgent in a Jenga environment.

    The agent interacts with the environment over a series of episodes. If `if_training_against_adversary` is True,
    the agent will train against an adversary initialized with the given strategy.

    Args:
        num_episodes (int): Number of episodes to run for.
        agent_simulations (int): Number of MCTS simulations per action selection.
        exploration_weight (float): Exploration weight for UCB1.
        if_training_against_adversary (bool): Whether to train against an adversary.
        strategy (Strategy): Strategy for the adversary to take.
        efficiency_threshold (int): The minimum number of moves before the tower falls to consider the strategy
                                    efficient.
    """
    print("Starting a new MCTS training loop")

    env = Environment()
    mcts_agent = MCTSAgent(env, num_simulations=agent_simulations, exploration_weight=exploration_weight)

    adversary = None
    if if_training_against_adversary:
        adversary = Adversary(strategy)
        print(f"The agent is training against an adversary with {strategy.__class__.__name__} strategy")

    for episode in range(1, num_episodes + 1):
        print(f"Started episode {episode} out of {num_episodes}")
        env.reset()  # Reset the environment for a new episode
        taken_actions = set()  # Reset the made actions
        previous_action = None
        state = get_state_from_image(env.get_screenshot())  # Get and preprocess the initial state
        move_count = 0  # Track the number of moves in the current episode
        players = [(mcts_agent, "Agent"), (adversary, "Adversary")]

        for player, role in itertools.cycle(players):
            if player is None:  # Skip if there's no adversary
                continue

            if adversary:  # If there is an adversary, log whose move it is
                print(f"{role}'s move")

            action = player.select_action(state, taken_actions)

            if action is None:
                print("No action to take. Ending the episode")
                break

            screenshot, is_fallen = env.step(action)
            state = get_state_from_image(screenshot)
            taken_actions.add(action)
            move_count += 1

            if is_fallen:
                print("The tower has fallen. Ending the episode")
                break

        # Adjust exploration if the tower fell too quickly
        if move_count < efficiency_threshold:
            print("Increasing exploration due to quick fall")
            exploration_weight *= 1.1
        else:
            print("Decreasing exploration due to sustained tower")
            exploration_weight *= 0.9


if __name__ == "__main__":
    # First phase: the MCTS agent trains against itself
    mcts_training_loop(if_training_against_adversary=False)

    # Second phase: the MCTS agent trains against the random-strategy adversary
    mcts_training_loop(if_training_against_adversary=True, strategy=RandomStrategy())

    # Third phase: the MCTS agent trains against a pessimistic-strategy adversary
    mcts_training_loop(if_training_against_adversary=True, strategy=PessimisticStrategy())

    # Fourth phase: the MCTS agent trains against an optimistic-strategy adversary
    mcts_training_loop(if_training_against_adversary=True, strategy=OptimisticStrategy())
