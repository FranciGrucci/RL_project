import argparse
import gymnasium as gym
from ddpg_agent import DDPG_Agent
import time
import config


def main():
    parser = argparse.ArgumentParser(description='Run training and evaluation')
    parser.add_argument('--render', default="human")
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('--evaluate', type=str)

    args = parser.parse_args()

    if args.train:

        train_dict = config.train_dict
        train_dict["eval"] = args.evaluate
        print(train_dict)

        agent = DDPG_Agent(**train_dict)
        agent.train(batch_size=config.BATCH_SIZE,checkpoint_frequency=config.CHECKPOINT_FREQUENCY)

    if args.evaluate:
        state_dim = 1
        action_dim = 1
        env = gym.make('Humanoid-v4', render_mode="human")

        for dim in env.observation_space.shape:
            state_dim *= dim
        print("State Dimension is:", state_dim)

        for dim in env.action_space.shape:
            action_dim *= dim
        print("Action Dimension is:", action_dim)

        max_action = float(env.action_space.high[0])

        agent = DDPG_Agent(state_dim, action_dim,
                           max_action, eval=True, env=env)
        agent.evaluate(env, checkpoint_path=args.evaluate)


if __name__ == '__main__':
    main()
