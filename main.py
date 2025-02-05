# import numpy as np
# import torch
# import gymnasium as gym
# import argparse
# from prova import DDPG_Agent  # Assumendo che il codice DDPG sia in un file chiamato ddpg.py

# # ---- Funzione per eseguire un episodio ----
# def run_episode(env, agent, train=True, max_steps=1000, render=False):
#     state, _ = env.reset()
#     state = np.array(state, dtype=np.float32).flatten()
#     episode_reward = 0

#     for step in range(max_steps):
#         if render:
#             env.render()

#         action = agent.select_action(state, noise=0.1 if train else 0.0)
#         next_state, reward, done, _, _ = env.step(action)
#         next_state = np.array(next_state, dtype=np.float32).flatten()

#         if train:
#             agent.replay_buffer.push(state, action, reward, next_state, done)
#             agent.train()

#         state = next_state
#         episode_reward += reward

#         if done:
#             break

#     return episode_reward

# # ---- Funzione principale ----
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--train", action="store_true", help="Train the agent")
#     parser.add_argument("--test", action="store_true", help="Test the trained agent")
#     parser.add_argument("--episodes", type=int, default=500, help="Number of episodes")
#     parser.add_argument("--render", action="store_true", help="Render the environment")
#     parser.add_argument("--load", type=str, default=None, help="Path to load the model")
#     parser.add_argument("--save", type=str, default="ddpg_model.pth", help="Path to save the model")
#     args = parser.parse_args()

#     # ---- Configura ambiente ----
#     env = gym.make("CarRacing-v2", continuous=True)
#     state_dim = env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2]
#     action_dim = env.action_space.shape[0]
#     max_action = float(env.action_space.high[0])

#     # ---- Inizializza agente ----
#     agent = DDPG_Agent(state_dim, action_dim, max_action)

#     # ---- Carica il modello, se specificato ----
#     if args.load:
#         agent.actor.load_state_dict(torch.load(args.load))
#         agent.critic.load_state_dict(torch.load(args.load.replace("actor", "critic")))
#         print(f"Modello caricato da {args.load}")

#     # ---- Modalità Addestramento ----
#     if args.train:
#         for episode in range(args.episodes):
#             reward = run_episode(env, agent, train=True, render=args.render)
#             print(f"Episode {episode + 1}/{args.episodes}, Reward: {reward}")

#         # Salva il modello dopo l'addestramento
#         torch.save(agent.actor.state_dict(), args.save)
#         torch.save(agent.critic.state_dict(), args.save.replace("actor", "critic"))
#         print(f"Modello salvato in {args.save}")

#     # ---- Modalità Test ----
#     if args.test:
#         total_rewards = []
#         for episode in range(10):  # Testa per 10 episodi
#             reward = run_episode(env, agent, train=False, render=args.render)
#             total_rewards.append(reward)
#             print(f"Test Episode {episode + 1}: Reward {reward}")

#         print(f"Media Reward Test: {np.mean(total_rewards)}")

#     env.close()

# if __name__ == "__main__":
#     main()




# # import argparse
# # import numpy as np
# # import gymnasium as gym
# # from function import Policy
# # import Actorcritic
# # import os
# # import pathlib

# # import matplotlib.pyplot as plt

# # def evaluate(env_name="CarRacing-v2", n_episodes=10, render=False):
# #     """
# #     Evaluate a trained agent on the specified environment.
# #     """
# #     agent = Policy()
# #     agent.load()

# #     env = gym.make(env_name, continuous=agent.continuous, render_mode="human" if render else None)
    
# #     rewards = []
# #     for episode in range(n_episodes):
# #         total_reward = 0
# #         done = False
# #         s, _ = env.reset()
        
# #         while not done:
# #             action = agent.act(s)
# #             s, reward, terminated, truncated, _ = env.step(action)
# #             done = terminated or truncated
# #             total_reward += reward
        
# #         rewards.append(total_reward)
# #         print(f"Episode {episode + 1}: Reward = {total_reward}")

# #     print('Mean Reward:', np.mean(rewards))


# # def train(n_episodes=1000):
# #     """
# #     Train the agent and save the model.
# #     """
# #     print("Calling train()")  # Debug
# #     agent = Policy()
# #     print("Policy initialized")
# #     agent.train()
    
# #     agent.save()
# #     print("Model saved successfully!")
# #     agent = Policy()
# #     score_history = []
# #     img_path = 'images'
# #     if not os.path.exists(img_path):
# #         pathlib.Path(img_path).mkdir(parents=True, exist_ok=True)
    
# #     for episode in range(n_episodes):
# #         done = False
# #         score = 0
# #         s, _ = gym.make('Humanoid-v5', continuous=agent.continuous).reset()
        
# #         while not done:
# #             action = agent.act(s)
# #             s, reward, terminated, truncated, info = gym.make('Humanoid-v5').step(action)
# #             done = terminated or truncated
# #             score += reward
# #             agent.learn()
        
# #         score_history.append(score)
# #         print(f'Episode {episode}, Score: {score}, Last 100 Avg: {np.mean(score_history[-100:])}')
        
# #         if episode % 50 == 0:
# #             agent.save()
# #             plt.plot(score_history)
# #             plt.xlabel('Episodes')
# #             plt.ylabel('Score')
# #             plt.grid()
# #             plt.savefig(os.path.join(img_path, "score_fig.png"))


# # def main():
# #     """
# #     Parses command-line arguments and runs training or evaluation.
# #     """
# #     parser = argparse.ArgumentParser(description='Run training and evaluation for CarRacing-v2 agent')
# #     parser.add_argument('-t', '--train', action='store_true', help='Train the agent')
# #     parser.add_argument('-e', '--evaluate', action='store_true', help='Evaluate the agent')
# #     parser.add_argument('-r', '--render', action='store_true', help='Render the evaluation')
# #     parser.add_argument('-n', '--episodes', type=int, default=1, help='Number of episodes for evaluation')

# #     args = parser.parse_args()

# #     if args.train:
# #         train()

# #     if args.evaluate:
# #         evaluate(n_episodes=args.episodes, render=args.render)

    
# # if __name__ == '__main__':
# #     main()






















# # import argparse
# # import os
# # import pathlib
# # import random
# # import numpy as np
# # import imageio
# # import matplotlib.pyplot as plt
# # import gymnasium as gym
# # from function import Policy


# # def evaluate(env=None, n_episodes=1, render=False):
# #     agent = Policy()
# #     agent.load()

# #     env = gym.make('Humanoid-v5', continuous=agent.continuous)
# #     if render:
# #         env = gym.make('Humanoid-v5', render_mode='human')
    
# #     rewards = []
# #     img_path = 'images'
# #     if not os.path.exists(img_path):
# #         pathlib.Path(img_path).mkdir(parents=True, exist_ok=True)
    
# #     for episode in range(n_episodes):
# #         total_reward = 0
# #         done = False
# #         s, _ = env.reset()
# #         frame_set = []
        
# #         while not done:
# #             action = agent.act(s)
# #             action = np.clip(action, -0.4, 0.4)
# #             s, reward, terminated, truncated, info = env.step(action)
# #             done = terminated or truncated
# #             total_reward += reward
            
# #             if render:
# #                 frame_set.append(env.render())
        
# #         if render:
# #             imageio.mimsave(os.path.join(img_path, f'eps-{episode}.gif'), frame_set, fps=30)
        
# #         rewards.append(total_reward)
# #         print(f'Episode {episode}, Reward: {total_reward}')
    
# #     print('Mean Reward:', np.mean(rewards))


# # def train(n_episodes=1000):
    
# #     print("Calling train()")  # Debug
# #     agent = Policy()
# #     print("Policy initialized")  # Debug
# #     agent.train()
# #     agent.save()

# #     agent = Policy()
# #     score_history = []
# #     img_path = 'images'
# #     if not os.path.exists(img_path):
# #         pathlib.Path(img_path).mkdir(parents=True, exist_ok=True)
    
# #     for episode in range(n_episodes):
# #         done = False
# #         score = 0
# #         s, _ = gym.make('Humanoid-v5', continuous=agent.continuous).reset()
        
# #         while not done:
# #             action = agent.act(s)
# #             s, reward, terminated, truncated, info = gym.make('Humanoid-v5').step(action)
# #             done = terminated or truncated
# #             score += reward
# #             agent.learn()
        
# #         score_history.append(score)
# #         print(f'Episode {episode}, Score: {score}, Last 100 Avg: {np.mean(score_history[-100:])}')
        
# #         if episode % 50 == 0:
# #             agent.save()
# #             plt.plot(score_history)
# #             plt.xlabel('Episodes')
# #             plt.ylabel('Score')
# #             plt.grid()
# #             plt.savefig(os.path.join(img_path, "score_fig.png"))


# # def main():
# #     parser = argparse.ArgumentParser(description='Run training and evaluation')
# #     parser.add_argument('--render', action='store_true')
# #     parser.add_argument('-t', '--train', action='store_true')
# #     parser.add_argument('-e', '--evaluate', action='store_true')
# #     args = parser.parse_args()

# #     if args.train:
# #         train()
# #     if args.evaluate:
# #         evaluate(render=args.render)


# # if __name__ == '__main__':
# #     main()


import argparse
import numpy as np
import torch
import gymnasium as gym
from prova import DDPG_Agent 

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

def train(n_episodes=2):
    """
    Addestra l'agente sull'ambiente CarRacing-v2 e salva il modello.
    """
    agent = DDPG_Agent()
    env = gym.make("CarRacing-v2", continuous=True)
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32).flatten()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, noise=0.1)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = np.array(next_state, dtype=np.float32).flatten()
            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.train()
            
            state = next_state
            total_reward += reward
            done = terminated or truncated
        
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
