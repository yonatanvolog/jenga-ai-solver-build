import time
from environment import \
    Environment  # Assuming the Environment class is in environment.py


def main():
    env = Environment()

    while True:
        has_fallen = env.is_fallen()
        print(f"Is the tower fallen? {has_fallen}")
        time.sleep(1)


if __name__ == "__main__":
    main()
