# Jenga AI Solver: Agents for Strategic Block Removal

This project focuses on developing AI agents capable of playing the game of **Jenga** using reinforcement learning and simulation-based strategies. The game involves players strategically removing blocks from a tower and placing them on top without causing the tower to collapse. Our solution models this problem in a **Unity-based simulation environment**, where agents remove blocks while aiming to destabilize the tower for their opponent.

## Agents
The project consists of three AI agents, each using a different approach to solve the problem:

- **GSBAS (Greedy Simulation-Based Action Search):**  
  This agent simulates multiple actions in real-time and chooses the one that maximizes the immediate reward based on the stability of the tower.
  
- **Hierarchical DQN:**  
  A Deep Q-Network agent that uses two separate networks to independently choose the **level** and **color** of the block to remove. This approach allows the agent to optimize both choices through reinforcement learning.
  
- **Hierarchical SARSA:**  
  A more cautious reinforcement learning agent that updates its Q-values based on the actions it actually took, not the optimal ones, leading to safer strategies during gameplay.

## Features
- **Strategic Block Removal:** Agents remove blocks while avoiding tower collapse, balancing risk and reward with each move.
- **Simulation-Based Learning:** Agents interact with a Unity physics engine that simulates realistic block collisions and tower instability.
- **Hierarchical Decision-Making:** The DQN and SARSA agents use separate neural networks to handle multi-level decisions, choosing the best block to remove based on learned strategies.

## Gameplay
In addition to solving the strategic Jenga problem, we've created a **playable game** that features all the agents except GSBAS and allows:
- **Two users** to play against each other.
- A user to play against one of our **AI agents**.
- Observing **AI agents** playing against each other to see how they handle the game.

Please note that all the following you can do on a Windows machine only!

## How to play
1. Go to prod_environment and run the only .exe file that is there.
2. From the root of the project, run game.py.
3. Choose your players and the number of rounds in the menu.
4. Click on the start button and enjoy! You can exit the game when it finishes or if you click on the Esc button.

## How to interact with the environment via CLI
1. Go to dev_environment and run the only .exe file that is there.
2. Run environment.py that is located in the same folder.

## How to train any of the agents
1. Go to the folder of the agent you want.
2. Run training_loop.py. The Jenga development build should run automatically. If they fail to connect, simply rerun 
   training_loop.py.

## How to see the performance of any of the agents on a graph?
1. Go to the folder of the agent you want.
2. Run plot.py (there are 1 for the GSBAS and 3 for the other agents plots available, uncomment the plot you want to run). 
   The Jenga development build should run automatically. If they fail to connect, simply rerun plot.py.

Feel free to explore the code, experiment with the agents, and try out the interactive game! If you'd like to know more about the project, check out the file report-jenga.pdf.
