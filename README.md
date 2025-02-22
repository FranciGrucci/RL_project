# Walking Humanoid with DDPG

This repository contains the project for the Reinforcement Learning exam, implementing a Deep Deterministic Policy Gradient (DDPG) agent for controlling a walking humanoid.

## Files Overview

### `config.py`
This file is used to configure all parameters for training and evaluation. You can edit:

- Training hyperparameters
- Noise type (Gaussian or Ornstein-Uhlenbeck)
- Replay buffer settings (with or without Prioritized Experience Replay - PER)
  
For further details, refer to the source code and comments within.


### `actor_critic.py`
Defines the Actor-Critic model classes. It includes predefined models for:

- Inverted Pendulum
- Hopper
- Humanoid

To use a specific model, uncomment the corresponding section and comment out the others.

### `ddpg_agent.py`
Contains the implementation of the DDPG agent, along with its:

- Training function
- Evaluation function

## Usage

### Installation
Before running the project, install the required dependencies:

```sh
pip install -r requirements.txt
```

### Training
To train the model, run:

```sh
python main.py --train
```

### Evaluation
To evaluate a trained model, use:

```sh
python main.py --evaluate <checkpoint_path>
```
Replace `<checkpoint_path>` with the path to the saved model checkpoint.


