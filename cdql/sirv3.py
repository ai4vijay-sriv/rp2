import gymnasium as gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime, date
from dataclasses import dataclass
import tyro
from pprint import pformat
import random



@dataclass
class Args:
    env_id: str = "MountainCar-v0"
    code_for: str = "coupled_lfa"
    alpha: float = 0.0001
    beta: float = 0.01
    gamma: float = 0.99
    epsilon_start: float = 1
    epsilon_final: float = 0.05
    exploration_fraction: float = 0.5
    num_episodes: int = 5000
    l: int = 2
    log_base_dir: str = "runs/lfa"
    today: str = date.today()
    eta = 0.125

# =================== CONFIG ===================
args = tyro.cli(Args)
env = gym.make(args.env_id)
n_actions = env.action_space.n
obs_dim = env.observation_space.shape[0]
log_name = f"{args.code_for}_{args.env_id}_rbf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
writer = SummaryWriter(log_dir=f"{args.log_base_dir}/{log_name}")
writer.add_text("Args", pformat(vars(args)), 0)


# Create RBF centers (l x l grid in state space)
pos_min, pos_max = env.observation_space.low[0], env.observation_space.high[0]
vel_min, vel_max = env.observation_space.low[1], env.observation_space.high[1]

pos_centers = np.linspace(pos_min + 0.5*((pos_max-pos_min)/args.l), pos_max - 0.5*((pos_max-pos_min)/args.l), args.l)
vel_centers = np.linspace(vel_min + 0.5*((vel_max-vel_min)/args.l), vel_max - 0.5*((vel_max-vel_min)/args.l), args.l)
rbf_centers = np.array([(p, v) for p in pos_centers for v in vel_centers])  # l^2 centers
pos_grid, vel_grid = np.meshgrid(pos_centers, vel_centers)

# Flatten to (N_centers, 2) array
rbf_centers = np.column_stack([pos_grid.ravel(), vel_grid.ravel()])

# RBF bandwidth (σ)
# sigma_pos = (pos_max - pos_min) / (args.l - 1) / 1.5
# sigma_vel = (vel_max - vel_min) / (args.l - 1) / 1.5

sigma_pos = args.eta * 1.8
sigma_vel = args.eta * 0.14

# Feature vector: RBFs over state
def phi(state):
    state = np.array(state).flatten()
    diffs = rbf_centers - state
    exponents = -((diffs[:, 0] ** 2) / (2 * sigma_pos ** 2) + (diffs[:, 1] ** 2) / (2 * sigma_vel ** 2))
    return np.exp(exponents).astype(np.float32)

feature_dim = len(phi(env.reset()[0]))
weights_u = np.zeros((n_actions, feature_dim), dtype=np.float32)
weights_v = np.zeros((n_actions, feature_dim), dtype=np.float32)
weights_w1 = np.zeros((n_actions, feature_dim), dtype=np.float32)
weights_w2 = np.zeros((n_actions, feature_dim), dtype=np.float32)


# ε-greedy schedule
def get_epsilon(episode):
    progress = min(episode / (args.exploration_fraction * args.num_episodes), 1.0)
    return args.epsilon_start + (args.epsilon_final - args.epsilon_start) * progress

# Q-value: dot product between action weight and RBF feature vector
def q_value_u(state, action):
    return np.dot(weights_u[action], phi(state))

def q_value_v(state, action):
    return np.dot(weights_v[action], phi(state))

def q_value_w1(state, action):
    return np.dot(weights_w1[action], phi(state))

def q_value_w2(state, action):
    return np.dot(weights_w2[action], phi(state))


# Policy: ε-greedy
def select_action(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    q_valsv = [q_value_v(state, a) for a in range(n_actions)]
    q_valsw2 = [q_value_w2(state, a) for a in range(n_actions)]
    q_sum = np.add(q_valsv, q_valsw2)  # element-wise sum
    return int(np.argmax(q_sum))


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



        if np.random.rand() < 0.5:  # Choose between two updates
            # Q-learning target
            next_action = np.argmax([q_value_w1(next_state, a) for a in range(n_actions)])
            q_next = q_value_u(next_state, next_action)
            td_target = reward + args.gamma * q_next
            td_error = td_target - q_value_v(state, action)

            weights_u[action] += args.alpha * ((phi(state) * q_value_v(state, action)) - weights_u[action])
            weights_v[action] += args.beta * td_error * phi(state)
            


        else:
            next_action = np.argmax([q_value_u(next_state, a) for a in range(n_actions)])
            q_next = q_value_w1(next_state, next_action)
            td_target = reward + args.gamma * q_next
            td_error = td_target - q_value_w2(state, action)

            weights_w1[action] += args.alpha * ((phi(state) * q_value_w2(state, action)) - weights_w1[action])
            weights_w2[action] += args.beta * td_error * phi(state)          

        state = next_state

    # Log to TensorBoard
    writer.add_scalar("Reward", total_reward, episode)
    writer.add_scalar("Epsilon", epsilon, episode)
    writer.add_scalar("weight_1_u", weights_u[0].mean(), episode)
    writer.add_scalar("weight_2_u", weights_u[1].mean(), episode)     
    writer.add_scalar("weight_3_u", weights_u[2].mean(), episode)
    writer.add_scalar("weight_1_v", weights_v[0].mean(), episode)
    writer.add_scalar("weight_2_v", weights_v[1].mean(), episode)
    writer.add_scalar("weight_3_v", weights_v[2].mean(), episode)
    writer.add_scalar("weight_1_w1", weights_w1[0].mean(), episode)
    writer.add_scalar("weight_2_w1", weights_w1[1].mean(), episode)
    writer.add_scalar("weight_3_w1", weights_w1[2].mean(), episode)
    writer.add_scalar("weight_1_w2", weights_w2[0].mean(), episode)
    writer.add_scalar("weight_2_w2", weights_w2[1].mean(), episode)
    writer.add_scalar("weight_3_w2", weights_w2[2].mean(), episode)
    print(f"Episode {episode+1}/{args.num_episodes} | Total reward: {total_reward:.1f}")


env.close()
writer.close()
