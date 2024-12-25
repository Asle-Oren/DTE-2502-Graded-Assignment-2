# Snake Reinforcement Learning (PyTorch)

This repository contains pytorch code based on the tensorflow implementation found here: https://github.com/DragonWarrior15/snake-rl
There is only pytorch code for the DeepQLearningAgent

Code for training a Deep Reinforcement Learning agent to play the game of Snake.
The agent takes 2 frames of the game as input (image) and predicts the action values for
the next action to take.
***
Sample games from the best performing [agent](../models/v15.1/model_188000.h5)<br>
<img width="400" height="400" src="https://github.com/Asle-Oren/DTE-2502-Graded-Assignment-2/blob/main/images/game_visual_v17.1_163500_14_ob_0.mp4" alt="model v17.1 agent" ><img width="400" height="400" src="https://github.com/Asle-Oren/DTE-2502-Graded-Assignment-2/blob/main/images/game_visual_v17.1_163500_14_ob_1.mp4" alt="model v17.1 agent" >
<img width="400" height="400" src="https://github.com/Asle-Oren/DTE-2502-Graded-Assignment-2/blob/main/images/game_visual_v17.1_163500_14_ob_2.mp4" alt="model v17.1 agent" ><img width="400" height="400" src="https://github.com/Asle-Oren/DTE-2502-Graded-Assignment-2/blob/main/images/game_visual_v17.1_163500_14_ob_3.mp4" alt="model v17.1 agent" >
***

# Running Graded assignment 02
To run this code do the following:
- Clone/download repo
- Make an enviroment using conda or .venv ex: python -m venv .venv
- Activate enviroment. ex: .venv\Scripts\activate or source .venv/bin/activate
- Install requirements. If using venv: pip install -r venvrequirements.txt
- Run training or skip to use existing model: python training.py
- Render videos of the model playing: python game_visualization.py

