import gymnasium as gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Environment
env = gym.make("MountainCar-v0")
n_actions = env.action_space.n
obs_dim = env.observation_space.shape[0]  # Should be 2 (position, velocity)

log_name = f"mountaincar_rbf_v2{datetime.now().strftime('%Y%m%d_%H%M%S')}"
writer = SummaryWriter(log_dir=f"runs/lfa/{log_name}")

# Hyperparameters
alpha = 0.01
beta = 0.01  
gamma = 0.99
epsilon_start = 0.5
epsilon_final = 0.05
exploration_fraction = 0.8
num_episodes = 50000

# Create RBF centers (l x l grid in state space)
l = 2  # You can adjust this (l=2 or l=3 are common)
pos_min, pos_max = env.observation_space.low[0], env.observation_space.high[0]
vel_min, vel_max = env.observation_space.low[1], env.observation_space.high[1]

pos_centers = np.linspace(pos_min, pos_max, l)
vel_centers = np.linspace(vel_min, vel_max, l)
rbf_centers = np.array([(p, v) for p in pos_centers for v in vel_centers])  # l^2 centers

# RBF bandwidth (σ)
sigma_pos = (pos_max - pos_min) / (l - 1) / 1.5
sigma_vel = (vel_max - vel_min) / (l - 1) / 1.5

# Feature vector: RBFs over state
def phi(state):
    state = np.array(state)
    diffs = rbf_centers - state
    exponents = -((diffs[:, 0] ** 2) / (2 * sigma_pos ** 2) + (diffs[:, 1] ** 2) / (2 * sigma_vel ** 2))
    return np.exp(exponents).astype(np.float32)

feature_dim = len(phi(env.reset()[0]))
weights_u = np.zeros((n_actions, feature_dim), dtype=np.float32)
weights_v = np.zeros((n_actions, feature_dim), dtype=np.float32)


# ε-greedy schedule
def get_epsilon(episode):
    progress = min(episode / (exploration_fraction * num_episodes), 1.0)
    return epsilon_start + (epsilon_final - epsilon_start) * progress

# Q-value: dot product between action weight and RBF feature vector
def q_value_u(state, action):
    return np.dot(weights_u[action], phi(state))

def q_value_v(state, action):
    return np.dot(weights_v[action], phi(state))


# Policy: ε-greedy
def select_action(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    q_vals = [q_value_v(state, a) for a in range(n_actions)]
    return int(np.argmax(q_vals))


for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    epsilon = get_epsilon(episode)

    while not done:
        action = select_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Q-learning target
        q_next = max(q_value_u(next_state, a) for a in range(n_actions))
        td_target = reward + gamma * q_next
        td_error = td_target - q_value_v(state, action)

        # Gradient update
        weights_u[action] += alpha * ((phi(state) * q_value_v(state, action)) - weights_u[action])
        weights_v[action] += beta * td_error * phi(state)

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
    print(f"Episode {episode+1}/{num_episodes} | Total reward: {total_reward:.1f}")


env.close()
writer.close()
