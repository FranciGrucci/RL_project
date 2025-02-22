from replay_buffer import ReplayBuffer, Experience_replay_buffer
#from exp_replay_buff import Experience_replay_buffer
import torch
from ornsteinuhlebeck import OrnsteinUhlenbeckNoise
import gymnasium as gym

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")


def compute_dimension(env):
    state_dim = 1
    action_dim = 1

    for dim in env.observation_space.shape:
        state_dim *= dim

    for dim in env.action_space.shape:
        action_dim *= dim

    return {"state_dim": state_dim, "action_dim": action_dim}

################### [TRAIN]##############################################


ENV_NAME = 'Humanoid-v4'
ENV = gym.make(ENV_NAME)
DIMENSIONS = compute_dimension(ENV)
MAX_ACTION = float(ENV.action_space.high[0])

TAU = 0.05
GAMMA = 0.99
CRITIC_LR = 1e-4
ACTOR_LR = 1e-4
BATCH_SIZE = 64
MU = 0
SIGMA = 0.2
THETA = 0.15
DT = 0.01
X0 = None
NOISE = OrnsteinUhlenbeckNoise(
    action_dim=DIMENSIONS["action_dim"], mu=MU, sigma=SIGMA, theta=THETA, dt=DT, x0=None)  # 0.1
CHECKPOINT_FREQUENCY = 50
################### [BUFFER]##############################################
# Uncomment the required buffer and comment the other

MEMORY_SIZE = 1000
BURN_IN = 90

# replay_buffer = ReplayBuffer(device=device,max_size=memory_size,burn_in = burn_in)

ALPHA = 0.6
BETA = 0.4
BETA_UPDATE_FREQUENCY = 100

replay_buffer = Experience_replay_buffer(
    device=DEVICE, memory_size=MEMORY_SIZE, burn_in=BURN_IN, alpha=ALPHA, beta=BETA,beta_update_frequency=BETA_UPDATE_FREQUENCY)
###########################################################################

train_dict = {

    "env": ENV,
    "state_dim": DIMENSIONS["state_dim"],
    "action_dim": DIMENSIONS["action_dim"],
    "max_action": MAX_ACTION,
    "noise": NOISE,
    "replay_buffer": replay_buffer,
    "critic_lr": CRITIC_LR,
    "actor_lr": ACTOR_LR,
    "gamma": GAMMA,
    "tau": TAU,
    "device": DEVICE

}
