import argparse
import gymnasium as gym
from ddpg_agent import DDPG_Agent
import config


def print_dict(dict):
    for key in dict:
        print(f"{key}: {dict[key]}")
    print()


def main():
    parser = argparse.ArgumentParser(description='Run training and evaluation')
    parser.add_argument('--render', default="human")
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-e', '--evaluate', type=str)

    args = parser.parse_args()

    if args.train:

        train_dict = config.train_dict
        train_dict["eval"] = False
        print_dict(train_dict)

        agent = DDPG_Agent(**train_dict)
        agent.train(batch_size=config.BATCH_SIZE,
                    checkpoint_frequency=config.CHECKPOINT_FREQUENCY,mean_reward_threshold=config.MEAN_REWARD_THRESHOLD,window=config.WINDOW)

    if args.evaluate:
        eval_dict = config.train_dict
        eval_dict["eval"] = True
        eval_dict["env"] = gym.make(config.ENV_NAME, render_mode="human")

        agent = DDPG_Agent(**eval_dict)

        agent.evaluate(eval_dict["env"], checkpoint_path=args.evaluate)


if __name__ == '__main__':
    main()
