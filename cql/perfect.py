import gymnasium as gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from dataclasses import dataclass
import tyro
from pprint import pformat
import os
import random


@dataclass
class Args:
    env_id: str = "MountainCar-v0"
    alpha: float = 0.01
    gamma: float = 0.99
    epsilon_start: float = 0.5
    epsilon_final: float = 0.05
    exploration_fraction: float = 0.3
    num_episodes: int = 500
    l: int = 2
    log_base_dir: str = "runs/lfa"

# =================== CONFIG ===================
args = tyro.cli(Args)
filename = os.path.splitext(os.path.basename(__file__))[0]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_name = (
    f"{filename}_"
    f"{args.env_id}_"
    f"{args.exploration_fraction:.2f}_"
    f"{args.num_episodes}_"
    f"{timestamp}"
)
# =================== ENV ======================
env = gym.make(args.env_id)
n_actions = env.action_space.n
obs_dim = env.observation_space.shape[0]

# =================== LOGGING ==================
#log_name = f"{args.env_id}_rbf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
writer = SummaryWriter(log_dir=f"{args.log_base_dir}/{log_name}")
writer.add_text("Args", pformat(vars(args)), 0)

# =================== RBF CENTERS ==============
pos_min, pos_max = env.observation_space.low[0], env.observation_space.high[0]
vel_min, vel_max = env.observation_space.low[1], env.observation_space.high[1]

pos_centers = np.linspace(pos_min, pos_max, args.l)
vel_centers = np.linspace(vel_min, vel_max, args.l)
rbf_centers = np.array([(p, v) for p in pos_centers for v in vel_centers])

sigma_pos = (pos_max - pos_min) / (args.l - 1) / 1.5
sigma_vel = (vel_max - vel_min) / (args.l - 1) / 1.5

# =================== FEATURE MAP ==============
def phi(state):
    state = np.array(state)
    diffs = rbf_centers - state
    exponents = -((diffs[:, 0] ** 2) / (2 * sigma_pos ** 2) + (diffs[:, 1] ** 2) / (2 * sigma_vel ** 2))
    return np.exp(exponents).astype(np.float32)

feature_dim = len(phi(env.reset()[0]))
weights = np.zeros((n_actions, feature_dim), dtype=np.float32)

# =================== EPSILON SCHEDULE =========
def get_epsilon(episode):
    progress = min(episode / (args.exploration_fraction * args.num_episodes), 1.0)
    return args.epsilon_start + (args.epsilon_final - args.epsilon_start) * progress

# =================== Q-VALUE ==================
def q_value(state, action):
    return np.dot(weights[action], phi(state))

# =================== POLICY ===================
def select_action(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    q_vals = [q_value(state, a) for a in range(n_actions)]
    return int(np.argmax(q_vals))

# =================== TRAIN LOOP ===============
for episode in range(args.num_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    epsilon = get_epsilon(episode)

    while not done:
        action = select_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Q-learning update
        q_next = max(q_value(next_state, a) for a in range(n_actions))
        td_target = reward + args.gamma * q_next
        td_error = td_target - q_value(state, action)
        weights[action] += args.alpha * td_error * phi(state)

        state = next_state

    # Log to TensorBoard
    writer.add_scalar("Reward", total_reward, episode)
    writer.add_scalar("Epsilon", epsilon, episode)
    for i in range(n_actions):
        writer.add_scalar(f"weight_{i+1}", weights[i].mean(), episode)

    print(f"Episode {episode+1}/{args.num_episodes} | Total reward: {total_reward:.1f}")

env.close()
writer.close()
