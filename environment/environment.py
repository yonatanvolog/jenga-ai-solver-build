import socket
import os
import time
import subprocess
from enum import Enum
import threading

from jinja2.ext import loopcontrols

MAX_LEVEL = 12
MAX_BLOCKS_IN_LEVEL = 3
SCREENSHOT_SHAPE = (128, 64)
# Mapping from integer to color for Jenga blocks
INT_TO_COLOR = {0: "y", 1: "b", 2: "g"}
COLOR_TO_INT = {"y": 0, "b": 1, "g": 2}


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

class CommandType(Enum):
    REMOVE = "remove"
    RESET = "reset"
    TIMESCALE = "timescale"
    ISFALLEN = "isfallen"
    SETSTATICFRICTION = "staticfriction"
    SETDYNAMICFRICTION = "dynamicfriction"
    SETSCREENSHOTRES = "set_screenshot_res"
    SETCOLLIDERDISTANCE = "set_fall_detect_distance"
    GETNUMOFBLOCKSINLEVEL = "get_num_of_blocks_in_level"
    GETAVERAGEMAXTILTANGLE = "get_average_max_tilt_angle"
    GETMOSTMAXTILTANGLE = "get_most_max_tilt_angle"  # Added new command
    REVERTSTEP = "revert_step"
    TOGGLEMENU = "toggle_menu"  # Added new command for toggle menu
    UNKNOWN = "unknown"


class Environment:
    def __init__(self, host="127.0.0.1", port_receive=25001, port_send=25002,
                 unity_exe_path="../environment/jenga-game.exe",
                 relative_path_to_screenshots="../environment/screenshots"):
        """Initialize the Environment with the host, port for receiving, port for sending, and path to the Unity
        executable."""
        self.host = host
        self.port_receive = port_receive
        self.port_send = port_send
        self.unity_exe_path = unity_exe_path
        self.relative_path_to_screenshots = relative_path_to_screenshots
        self.unity_process = None
        self.last_action = None  # To store the last action for reverting

        if self.unity_exe_path:
            try:
                self.start_unity()
            except FileNotFoundError:
                print(f"Warning: Unity executable not found at {self.unity_exe_path}. "
                      f"Continuing without launching Unity.")

        self.set_timescale(100)  # Speed up the simulation

    def __enter__(self):
        """Context management entry point."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context management exit point for cleanup."""
        self.cleanup()

    def cleanup(self):
        """Cleanup function to stop Unity when the Python script exits."""
        self.stop_unity()

    def start_unity(self):
        """Start the Unity executable."""
        if self.unity_exe_path and not self.unity_process:
            self.unity_process = subprocess.Popen([self.unity_exe_path])
            print(f"Started Unity from {self.unity_exe_path}")

    def stop_unity(self):
        """Stop the Unity executable."""
        if self.unity_process:
            self.unity_process.terminate()
            self.unity_process.wait()  # Ensure the process has terminated
            print("Unity process terminated.")
            self.unity_process = None

    def send_command(self, command, retry_attempts=3, retry_delay=0.5):
        """Send a command to the Unity environment and receive the response."""
        for attempt in range(retry_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.connect((self.host, self.port_receive))
                    sock.sendall(command.encode("utf-8"))
                    response = sock.recv(1024).decode("utf-8")
                    return response.strip()
            except (ConnectionResetError, ConnectionRefusedError) as e:
                print(f"Connection error: {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)

        raise ConnectionError(f"Failed to send command after {retry_attempts} attempts.")

    def listen_for_commands(self):
        """Listen for incoming commands from Unity and print them."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((self.host, self.port_send))  # Use the port_send (25002) for receiving commands
            sock.listen(1)
            print(f"Listening for incoming Unity commands on port {self.port_send}...")
            while True:
                client, addr = sock.accept()
                with client:
                    print(f"Connected to Unity at {addr}")
                    while True:
                        data = client.recv(1024)
                        if not data:
                            break
                        command = data.decode('utf-8').strip()
                        print(f"Received command from Unity: {command}")

                        return command

    def reset(self):
        """Reset the Jenga game immediately."""
        response = self.send_command("reset")
        self.last_action = None  # Resetting, so clear the last action
        return response

    def toggle_menu(self):
        """Send a command to toggle the menu in Unity."""
        response = self.send_command("toggle_menu")
        return response

    def select_action(self):
        response = self.send_command("human")
        return response  # tuple of format (level, color)

    def step(self, action, if_get_state=True, wait_time=0.5):
        """
        Perform an action to remove a piece from the Jenga tower.

        Parameters:
            action (tuple): A tuple containing the level (int) and color (str - 'y', 'g', 'b') of the piece to remove.
            if_get_state (bool): Signifies whether the screenshot needs to be taken.
            wait_time (float): Number of seconds from making the action till making a snapshot.

        Returns:
            tuple: A tuple containing the path to the screenshot (str) and a boolean indicating if the tower has fallen.
        """
        level, color = action
        command = f"remove {level} {color}"
        self.send_command(command)
        self.last_action = action  # Store the last action for possible reverting

        if not if_get_state:
            return

        # Check if the tower has fallen
        time.sleep(wait_time)
        is_fallen = self.is_fallen()

        # Retrieve the screenshot after performing the action
        time.sleep(wait_time)
        screenshot = self.get_screenshot()
        return screenshot, is_fallen

    def revert_step(self):
        """
        Revert the last action performed by the step method.

        This command returns the environment to the same condition it was before the last call to the "step" method.

        Returns:
            str: A confirmation message that the step has been reverted.
        """
        if self.last_action:
            level, color = self.last_action
            command = f"revert_step {level} {color}"
            response = self.send_command(command)
            self.last_action = None  # Clear the last action since we've reverted it
            return response
        else:
            return "No step to revert."

    def set_timescale(self, timescale):
        """
        Set the timescale of the Jenga game.

        Parameters:
            timescale (float): The timescale to set. For example, a timescale of 2 means the game runs twice as fast as
            real life.

        Note:
            This value is not saved between runs of the game and defaults to 1. You need to set it each time to ensure
            it is at the desired value.
        """
        command = f"timescale {timescale}"
        response = self.send_command(command)
        return response

    def set_static_friction(self, static_friction):
        """
        Set the static friction of the Jenga pieces.

        Parameters:
            static_friction (float): The static friction value to set.

        Note:
            This value is saved between runs of the game, so you need to set it to ensure it is at the desired value.
        """
        command = f"staticfriction {static_friction}"
        response = self.send_command(command)
        return response

    def set_dynamic_friction(self, dynamic_friction):
        """
        Set the dynamic friction of the Jenga pieces.

        Parameters:
            dynamic_friction (float): The dynamic friction value to set.

        Note:
            This value is saved between runs of the game, so you need to set it to ensure it is at the desired value.
        """
        command = f"dynamicfriction {dynamic_friction}"
        response = self.send_command(command)
        return response

    def set_screenshot_res(self, width):
        """
        Set the resolution width for screenshots taken in the Jenga game.

        Parameters:
            width (int): The width to set for the screenshot resolution.
        """
        command = f"set_screenshot_res {width}"
        response = self.send_command(command)
        return response

    def set_collider_distance(self, value):
        """
        Set the distance of the colliders around the Jenga tower.

        This method receives a float value (positive or negative), which will be added to the current distance of the
        colliders from the tower. When calling the method, a visual representation of the colliders' distance will be
        shown in the Unity game.

        Parameters:
            value (float): The value to adjust the colliders' distance.
        """
        command = f"set_fall_detect_distance {value}"
        response = self.send_command(command)
        return response

    def get_num_of_blocks_in_level(self, level):
        """
        Get the number of blocks in a specific level of the Jenga tower.

        This method receives an integer `level` (0 is the top) and returns the number of blocks in that level.

        Parameters:
            level (int): The level for which to retrieve the number of blocks.

        Returns:
            int: The number of blocks in the specified level.
        """
        command = f"get_num_of_blocks_in_level {level}"
        response = self.send_command(command)
        return int(response)

    def get_average_max_tilt_angle(self):
        """
        Get the average of the maximum tilt angles recorded for all cubes in the Jenga tower.

        The average returned represents the average of the maximum tilt angles experienced by the tower
        between the removal of cubes. After removing a cube, the tilt value for each remaining cube
        is reset to its current tilt angle. Essentially, this method shows the maximum tilt angle
        the tower has experienced after the last cube was removed.

        In order to get a precise result, you need to wait some time after the removal of a cube
        before checking the value returned by this method. If the tower has fallen, the value returned
        will be inaccurate and should be disregarded.
        """
        command = "get_average_max_tilt_angle"
        response = self.send_command(command)
        while not response:
            response = self.send_command(command)
        return float(response)

    def get_most_max_tilt_angle(self):
        """
        Get the maximum tilt angle recorded among all cubes in the Jenga tower.

        Returns:
            float: The maximum tilt angle.
        """
        command = "get_most_max_tilt_angle"
        response = self.send_command(command)
        while not response:
            response = self.send_command(command)
        return float(response)

    def is_fallen(self):
        """
        Check if the Jenga tower has fallen.

        Returns:
            bool: True if the tower has fallen, False otherwise.

        Note:
            If the tower is slowly falling but hasn't tilted enough yet, this method may return False.
            It's advised to call this method after some delay following an action.
        """
        response = self.send_command("isfallen")
        return response.lower() == "true"

    def get_screenshot(self, wait_time=1, retry_attempts=3, retry_delay=0.25):
        """
        Capture a screenshot of the Jenga tower from a 45-degree angle, showing two sides.

        Returns:
            str: The path to the screenshot file.

        Note:
            This method deletes any previous screenshot in the folder before saving the new one.
        """
        # Directory where screenshots are saved
        screenshot_dir = os.path.join(os.getcwd(), self.relative_path_to_screenshots)

        time.sleep(wait_time)

        # Find the only PNG file in the directory
        png_files = [f for f in os.listdir(screenshot_dir) if f.endswith('.png')]
        attempt_count = 1
        while len(png_files) < 1 and attempt_count < retry_attempts:
            time.sleep(retry_delay)
            png_files = [f for f in os.listdir(screenshot_dir) if f.endswith('.png')]
            attempt_count += 1

        if len(png_files) == 1:
            screenshot_path = os.path.join(screenshot_dir, png_files[0])
            return screenshot_path
        else:
            raise FileNotFoundError(
                "Expected one PNG file in the directory, but found {}".format(
                    len(png_files)))


# DONT REMOVE THIS METHOD!
def performance_test():
    env = Environment()

    for level in range(11):  # Levels from 0 to 11
        for color in ['y', 'b', 'g']:  # For each color
            env.step((level, color))


def start_game():
    """
    Main game loop to start the game and determine player types and number of rounds.
    """
    # TODO: change to environment/screenshots
    env = Environment(relative_path_to_screenshots="/screenshots", unity_exe_path=None)
    env.reset()
    env.toggle_menu()

    while True:
        start_command = env.listen_for_commands()
        if start_command.startswith("start"):
            _, p1_type, p2_type, num_of_rounds = start_command.split()
            num_of_rounds = int(num_of_rounds)
            p1_type = PlayerType(int(p1_type))
            p2_type = PlayerType(int(p2_type))
            start_two_player_game(env, num_of_rounds, (p1_type, p2_type))


def start_two_player_game(env, num_of_rounds, player_types):
    """
    Simulate a game between human and ai.

    Parameters:
        env: The game environment.
        num_of_rounds: The number of rounds the game should last.
    """
    current_round = 1
    current_player_index = 0
    players = [0, 1]  # Two players, indexed as 0 and 1

    print("Starting general 2 player game")
    while True:
        # Send command indicating it's the current player's turn
        print(f"Sending player_turn for Player {current_player_index}")

        env.send_command(
            f"player_turn {player_types[current_player_index].value} {current_player_index} {current_round}")

        if player_types[current_player_index] != PlayerType.HUMAN:
            simulate_ai_removing_piece(env, current_player_index)

        #if player_type== PlayerType.HUMAN, the player will remove piece via unity

        # Wait for the "finished_move" command from Unity
        command = env.listen_for_commands()
        print(f"Received command: {command}")
        if command.startswith("finished_move"):
            time.sleep(3)  # Wait for 3 seconds to simulate a pause, sleep does not freeze unity
            # MASHA: here you can get the screenshot
            # Check if the tower has fallen
            if env.is_fallen():
                print(f"Player {current_player_index} lost the game!")
                if current_round == num_of_rounds:
                    env.reset()
                    env.toggle_menu()
                    break
                else:
                    current_round += 1
                    env.reset()
                    continue

            # Alternate between player 0 and player 1
            current_player_index = 1 if current_player_index == 0 else 0

        if command.startswith("end_game"):
            env.reset()
            env.toggle_menu()
            return


def simulate_ai_removing_piece(env, player_index):
    """
    Simulate an AI player removing a piece from the Jenga tower.

    Parameters:
        env: The game environment.
        player_index: The index of the current player (0 for player 1, 1 for player 2).

    Note:
        Player 1 (p1) removes pieces in the order (level, 'y') from level 0 to 11.
        Player 2 (p2) removes pieces in the order (level, 'g') from level 0 to 11.
    """

    # Initialize counters for both players if they don't exist
    if not hasattr(simulate_ai_removing_piece, "p1_counter"):
        simulate_ai_removing_piece.p1_counter = 0
    if not hasattr(simulate_ai_removing_piece, "p2_counter"):
        simulate_ai_removing_piece.p2_counter = 0

    # Define the action sequence for p1 and p2
    p1_actions = [(level, 'y') for level in range(12)]  # p1 removes yellow pieces from level 0 to 11
    p2_actions = [(level, 'g') for level in range(12)]  # p2 removes green pieces from level 0 to 11

    # Execute the current action based on the player and their counter
    if player_index == 0:  # Player 1's turn
        if simulate_ai_removing_piece.p1_counter < len(p1_actions):
            action = p1_actions[simulate_ai_removing_piece.p1_counter]
            print(f"Player 1 (AI) removing piece at level {action[0]} with color {action[1]}")
            env.step(action, 3)
            simulate_ai_removing_piece.p1_counter += 1
        else:
            print("Player 1 (AI) has removed all available pieces.")

    elif player_index == 1:  # Player 2's turn
        if simulate_ai_removing_piece.p2_counter < len(p2_actions):
            action = p2_actions[simulate_ai_removing_piece.p2_counter]
            print(f"Player 2 (AI) removing piece at level {action[0]} with color {action[1]}")
            env.step(action, 3)
            simulate_ai_removing_piece.p2_counter += 1
        else:
            print("Player 2 (AI) has removed all available pieces.")


def main():
    unity_exe_path = os.path.join(os.getcwd(), "./jenga-game.exe")
    with Environment(unity_exe_path=unity_exe_path) as env:
        while True:
            print("\nWhat action would you like to do?")
            print("1: Reset Environment")
            print("2: Perform Action (Remove Piece)")
            print("3: Set Timescale")
            print("4: Set Static Friction")
            print("5: Set Dynamic Friction")
            print("6: Set Screenshot Resolution")
            print("7: Set Collider Distance")
            print("8: Get Number of Blocks in Level")
            print("9: Get Average Max Tilt Angle")
            print("10: Get Most Max Tilt Angle")  # Added menu item for new command
            print("11: Revert Last Step")
            print("12: Exit")

            choice = input("Enter the number of your choice: ").strip()

            if choice == "1":
                print("Resetting environment...")
                env.reset()
                print("Environment reset.")

            elif choice == "2":
                level = input("Enter the level number: ").strip()
                color = input("Enter the color (y, b, g): ").strip()
                wait_time = input("Enter the wait time in seconds (default is 0.5): ").strip()
                if wait_time:
                    wait_time = float(wait_time)
                else:
                    wait_time = 0.5
                action = (level, color)
                print(f"Performing action: remove piece at level {level}, color {color}...")
                screenshot, is_fallen = env.step(action, wait_time)
                print(f"Action performed. Screenshot saved at: {screenshot}")
                print(f"Has the tower fallen? {'Yes' if is_fallen else 'No'}")

            elif choice == "3":
                timescale = input("Enter the timescale value (e.g., 1.5): ").strip()
                print(f"Setting timescale to {timescale}...")
                env.set_timescale(float(timescale))
                print("Timescale set.")

            elif choice == "4":
                static_friction = input("Enter the static friction value: ").strip()
                print(f"Setting static friction to {static_friction}...")
                env.set_static_friction(float(static_friction))
                print("Static friction set.")

            elif choice == "5":
                dynamic_friction = input("Enter the dynamic friction value: ").strip()
                print(f"Setting dynamic friction to {dynamic_friction}...")
                env.set_dynamic_friction(float(dynamic_friction))
                print("Dynamic friction set.")

            elif choice == "6":
                width = input("Enter the screenshot resolution width: ").strip()
                if width.isdigit():
                    width = int(width)
                    print(f"Setting screenshot resolution width to {width}...")
                    env.set_screenshot_res(width)
                    print("Screenshot resolution set.")
                else:
                    print("Invalid width value.")

            elif choice == "7":
                distance = input("Enter the collider distance change value (e.g., -0.2 or 0.5): ").strip()
                print(f"Setting collider distance to {distance}...")
                env.set_collider_distance(float(distance))
                print("Collider distance set.")

            elif choice == "8":
                level = input("Enter the level number: ").strip()
                if level.isdigit():
                    num_of_blocks = env.get_num_of_blocks_in_level(int(level))
                    print(f"Number of blocks in level {level}: {num_of_blocks}")
                else:
                    print("Invalid level value.")

            elif choice == "9":
                print("Getting average max tilt angle...")
                average_tilt_angle = env.get_average_max_tilt_angle()
                print(f"Average max tilt angle: {average_tilt_angle}")

            elif choice == "10":
                print("Getting most max tilt angle...")  # Handle the new menu option
                most_max_tilt_angle = env.get_most_max_tilt_angle()
                print(f"Most max tilt angle: {most_max_tilt_angle}")

            elif choice == "11":
                print("Reverting last step...")
                response = env.revert_step()
                print(response)

            elif choice == "12":
                print("Exiting...")
                break

            else:
                print("Invalid choice, please try again.")


if __name__ == "__main__":
    start_game()
    #main()
    #test()
