class HumanAgent:
    def __init__(self, env):
        """
        Initializes the HumanAgent.

        Args:
            env (Environment): The Jenga game environment.
        """
        self.env = env  # Store the environment reference

    def select_action(self, state, taken_actions):
        """
        Selects an action based on human input.

        This function calls the `select_action` method of the environment, which prompts the human player
        for input.

        Args:
            state (torch.Tensor): The current state of the environment.
            taken_actions (Set[Tuple[int, int]]): Already performed actions.

        Returns:
            tuple: A tuple containing the selected level and color of the block to remove.
        """
        action = self.env.select_action()
        while action in taken_actions:
            action = self.env.select_action()
        return action
