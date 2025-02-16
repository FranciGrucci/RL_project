import argparse
import numpy as np
import torch
import gymnasium as gym
from ddpg_agent import DDPG_Agent
import time


def main():
    parser = argparse.ArgumentParser(description='Run training and evaluation')
    parser.add_argument('--render', default="human")
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('--evaluate', type=str)
    parser.add_argument('--n_episodes', type=int, default=10)

    args = parser.parse_args()
    if args.train:

        env = gym.make('Hopper-v4')

        state_dim = 1
        action_dim = 1

        for dim in env.observation_space.shape:
            state_dim *= dim
        print("State Dimension is:", state_dim)

        for dim in env.action_space.shape:
            action_dim *= dim
        print("Action Dimension is:", action_dim)

        max_action = float(env.action_space.high[0])

        agent = DDPG_Agent(state_dim=state_dim,
                           action_dim=action_dim, max_action=max_action, env=env, noise=0.1,memory_size=100000,burn_in=90000,alpha=0.6,beta=0.4)
        agent.train(n_episodes=args.n_episodes, batch_size=512)

    if args.evaluate:
        state_dim = 1
        action_dim = 1
        env = gym.make('Hopper-v4', render_mode="human")

        for dim in env.observation_space.shape:
            state_dim *= dim
        print("State Dimension is:", state_dim)

        for dim in env.action_space.shape:
            action_dim *= dim
        print("Action Dimension is:", action_dim)

        max_action = float(env.action_space.high[0])

        agent = DDPG_Agent(state_dim, action_dim,
                           max_action, eval=True, env=env)
        agent.evaluate(env,checkpoint_path=args.evaluate)


if __name__ == '__main__':
    main()
