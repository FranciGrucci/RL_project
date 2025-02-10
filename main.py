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
    
    env = gym.make("CarRacing-v2", continuous=True, render_mode="human" if render else None)
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

def train(n_episodes=10):
    """
    Addestra l'agente sull'ambiente CarRacing-v2 e salva il modello.
    """
    #agent = DDPG_Agent()
    env = gym.make("CarRacing-v2", continuous=True)
    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2]  # Flatten immagine
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = DDPG_Agent(state_dim, action_dim, max_action)
    for episode in range(n_episodes):
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32).flatten()
        #print(state.shape)
        total_reward = 0
        done = False
        while agent.replay_buffer.burn_in_capacity() < 1:
            #print(agent.replay_buffer.burn_in_capacity()*100)
            print("\rFull {:2f}%\t\t".format(agent.replay_buffer.burn_in_capacity()*100), end="")
                   
            #self.take_step(mode='explore')
            action = env.action_space.sample()
            s_1, r, terminated, truncated, _ = env.step(action)
            
            done = terminated or truncated
            #print("BURNIN",np.array(s_1, dtype=np.float32).flatten())
            agent.replay_buffer.push(state, action, r, done, np.array(s_1, dtype=np.float32).flatten())
            #print("PUSHATO")
            total_reward += r
            state = np.array(s_1, dtype=np.float32).flatten()
            
            if done:
               state, _ = env.reset()
               state = np.array(state, dtype=np.float32).flatten()
        
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32).flatten()
        #print(state.shape)
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, noise=0.1)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = np.array(next_state, dtype=np.float32).flatten()
            done = terminated or truncated
            agent.replay_buffer.push(state, action, reward, done, next_state)
            agent.train()
            
            state = next_state
            total_reward += reward
            if done:
               state, _ = env.reset()
               state = np.array(state, dtype=np.float32).flatten()
            #done = terminated or truncated
        
        print(f"Episodio {episode + 1}/{n_episodes}, Reward: {total_reward}")
    
    agent.save()
    env.close()

def main():
    parser = argparse.ArgumentParser(description='Run training and evaluation')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-e', '--evaluate', action='store_true')
   
    args = parser.parse_args()
    if args.train:
        train()
    
    if args.evaluate:
        evaluate(n_episodes=args.episodes, render=args.render)
    
if __name__ == '__main__':
    main()
