import socket
import os
import time
import subprocess
import atexit
import signal
from enum import Enum


class CommandType(Enum):
    REMOVE = "remove"
    RESET = "reset"
    TIMESCALE = "timescale"
    ISFALLEN = "isfallen"
    SETSTATICFRICTION = "staticfriction"
    SETDYNAMICFRICTION = "dynamicfriction"
    SETSCREENSHOTRES = "set_screenshot_res"
    UNKNOWN = "unknown"


class Environment:
    def __init__(self, host="127.0.0.1", port=25001, unity_exe_path=None):
        """Initialize the Environment with the host, port, and path to the Unity executable."""
        self.host = host
        self.port = port
        self.unity_exe_path = unity_exe_path
        self.unity_process = None

        if self.unity_exe_path:
            try:
                self.start_unity()
            except FileNotFoundError:
                print(f"Warning: Unity executable not found at {self.unity_exe_path}. Continuing without launching "
                      f"Unity.")

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
                    sock.connect((self.host, self.port))
                    sock.sendall(command.encode("utf-8"))
                    response = sock.recv(1024).decode("utf-8")
                    return response.strip()
            except (ConnectionResetError, ConnectionRefusedError) as e:
                print(f"Connection error: {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)

        raise ConnectionError(f"Failed to send command after {retry_attempts} attempts.")

    def reset(self):
        """Reset the Jenga game immediately."""
        response = self.send_command("reset")
        return response

    def step(self, action, wait_time=0.5):
        """
        Perform an action to remove a piece from the Jenga tower.

        Parameters:
            action (tuple): A tuple containing the level (int) and color (str - 'y', 'g', 'b') of the piece to remove.
            wait_time (float): Number of seconds from making the action till making a snapshot.

        Returns:
            tuple: A tuple containing the path to the screenshot (str) and a boolean indicating if the tower has fallen.
        """
        level, color = action
        command = f"remove {level} {color}"
        self.send_command(command)

        # Check if the tower has fallen
        is_fallen = self.is_fallen()

        # Retrieve the screenshot after performing the action
        time.sleep(wait_time)
        screenshot = self.get_screenshot()
        return screenshot, is_fallen

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

    @staticmethod
    def get_screenshot(retry_attempts=3, retry_delay=0.25):
        """
        Capture a screenshot of the Jenga tower from a 45-degree angle, showing two sides.

        Returns:
            str: The path to the screenshot file.

        Note:
            This method deletes any previous screenshot in the folder before saving the new one.
        """
        # Directory where screenshots are saved
        screenshot_dir = os.path.join(os.getcwd(), "..\screenshots")

        # Find the only PNG file in the directory
        png_files = [f for f in os.listdir(screenshot_dir) if f.endswith('.png')]
        attempt_count = 1
        while len(png_files) != 1 and attempt_count < retry_attempts:
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


def main():
    unity_exe_path = os.path.join(os.getcwd(), "../unity_build/jenga-game.exe")
    with Environment(unity_exe_path=unity_exe_path) as env:
        while True:
            print("\nWhat action would you like to do?")
            print("1: Reset Environment")
            print("2: Perform Action (Remove Piece)")
            print("3: Set Timescale")
            print("4: Set Static Friction")
            print("5: Set Dynamic Friction")
            print("6: Set Screenshot Resolution")
            print("7: Exit")

            choice = input("Enter the number of your choice: ").strip()

            if choice == "1":
                print("Resetting environment...")
                env.reset()
                print("Environment reset.")

            elif choice == "2":
                level = input("Enter the level number: ").strip()
                color = input("Enter the color (y, b, g): ").strip()
                wait_time = input(
                    "Enter the wait time in seconds (default is 0.5): ").strip()
                if wait_time:
                    wait_time = float(wait_time)
                else:
                    wait_time = 0.5
                action = (level, color)
                print(
                    f"Performing action: remove piece at level {level}, color {color}...")
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
                dynamic_friction = input(
                    "Enter the dynamic friction value: ").strip()
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
                print("Exiting...")
                break

            else:
                print("Invalid choice, please try again.")


if __name__ == "__main__":
    main()
