import random

import utils


class GSBASAgent:
    def __init__(self, env):
        """
        Initializes the GSBASAgent with the given environment.

        Args:
            env (Environment): The Jenga game environment.
        """
        self.env = env

    def select_action(self, state, taken_actions):
        """
        Selects the action with the maximum immediate reward by evaluating all possible actions from the current state.

        Args:
            state (torch.Tensor): The current state of the environment (not used in this agent, but left for
                                  compatibility with the DQN agent).
            taken_actions (Set[Tuple[int, int]]): Already performed actions.

        Returns:
            tuple: A tuple containing the selected level and color of the block to remove.
        """
        best_action = None
        best_reward = float('-inf')

        # Get all possible actions
        possible_actions = utils.get_possible_actions(taken_actions)
        previous_tilt = self.env.get_average_max_tilt_angle()

        for action in random.sample(possible_actions, min(10, len(possible_actions))):
            print(f"Simulating action {action}")

            # Simulate the action
            _, is_fallen = self.env.step(utils.format_action(action))
            if is_fallen:
                print("The tower has fallen while in simulation. Reverting")
                self.env.revert_step()
                continue

            # Get current tilt after action
            current_tilt = self.env.get_average_max_tilt_angle()

            # Calculate the reward for this action
            reward = utils.calculate_reward(action, previous_tilt, current_tilt)

            # Keep track of the best action
            if reward > best_reward:
                best_reward = reward
                best_action = action

            # Revert the action for the next evaluation
            self.env.revert_step()

        return best_action
