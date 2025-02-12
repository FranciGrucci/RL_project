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
    parser.add_argument('-e', '--evaluate', action='store_true')
    parser.add_argument('--n_episodes', type=int, default=10)

    args = parser.parse_args()
    if args.train:
        env = gym.make("CarRacing-v2", continuous=True)
        # Flatten immagine
        state_dim = env.observation_space.shape[0] * \
            env.observation_space.shape[1] * env.observation_space.shape[2]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        
        agent = DDPG_Agent(state_dim, action_dim, max_action, env=env)

        agent.train(n_episodes=args.n_episodes,batch_size=64)

    if args.evaluate:
        env = gym.make("CarRacing-v2", continuous=True,
                       render_mode=args.render)
        # Flatten immagine
        state_dim = env.observation_space.shape[0] * \
            env.observation_space.shape[1] * env.observation_space.shape[2]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

        agent = DDPG_Agent(state_dim, action_dim, max_action, env=env)

        agent.evaluate(env)


if __name__ == '__main__':
    main()
