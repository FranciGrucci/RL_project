import argparse
import numpy as np
import torch
import gymnasium as gym
from ddpg_agent import DDPG_Agent
import time


def evaluate(n_episodes=10, render=False):
    """
    Valuta un agente addestrato sull'ambiente CarRacing-v2.
    """
    agent = DDPG_Agent()
    agent.load()

    env = gym.make("CarRacing-v2", continuous=True,
                   render_mode="human" if render else None)
    rewards = []

    for episode in range(n_episodes):
        total_reward = 0
        done = False
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32).flatten()

        while not done:
            action = agent.select_action(state, noise=0.0)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = np.array(next_state, dtype=np.float32).flatten()

        rewards.append(total_reward)
        print(f"Episodio {episode + 1}: Reward = {total_reward}")

    print(f"Media Reward: {np.mean(rewards)}")
    env.close()


def main():
    parser = argparse.ArgumentParser(description='Run training and evaluation')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-e', '--evaluate', action='store_true')

    args = parser.parse_args()
    if args.train:
        env = gym.make("CarRacing-v2", continuous=True)
        # Flatten immagine
        state_dim = env.observation_space.shape[0] * \
            env.observation_space.shape[1] * env.observation_space.shape[2]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

        agent = DDPG_Agent(state_dim, action_dim, max_action,env=env)

        agent.train()

    if args.evaluate:
        evaluate(n_episodes=args.episodes, render=args.render)


if __name__ == '__main__':
    main()
