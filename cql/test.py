import gymnasium as gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# ===== Environment Setup =====
env = gym.make("MountainCar-v0")
pos_min, pos_max = env.observation_space.low[0], env.observation_space.high[0]
vel_min, vel_max = env.observation_space.low[1], env.observation_space.high[1]
n_actions = env.action_space.n

# ===== RBF Feature Construction =====
l = 2  # grid size (l x l)
pos_centers = np.linspace(pos_min, pos_max, l)
vel_centers = np.linspace(vel_min, vel_max, l)
centers = np.array([(p, v) for p in pos_centers for v in vel_centers])

sigma_pos = (pos_max - pos_min) / (l - 1) / 1.5
sigma_vel = (vel_max - vel_min) / (l - 1) / 1.5

def rbf_features(state):
    pos, vel = state
    diffs = centers - np.array([pos, vel])
    scaled_diffs = np.square(diffs / [sigma_pos, sigma_vel])
    activations = np.exp(-0.5 * np.sum(scaled_diffs, axis=1))
    return activations

def phi(state, action):
    base = rbf_features(state)
    feature = np.zeros(n_actions * len(base), dtype=np.float32)
    feature[action * len(base):(action + 1) * len(base)] = base
    norm = np.linalg.norm(feature)
    return feature / norm if norm > 0 else feature

feature_dim = n_actions * len(centers)

# ===== Coupled Q-Learning Initialization =====
u = np.zeros(feature_dim, dtype=np.float32)
v = np.zeros(feature_dim, dtype=np.float32)

alpha = 1e-4
beta = 1e-4
gamma = 0.99
epsilon = 0.3
num_episodes = 50000
max_steps = 200

log_name = f"mountaincar_cql_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
writer = SummaryWriter(log_dir=f"runs/lfa/{log_name}")

def Q(weights, state, action):
    return np.dot(weights, phi(state, action))

# ===== Training Loop =====
for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0

    for step in range(max_steps):
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            q_vals = [Q(v, state, a) for a in range(n_actions)]
            action = np.argmax(q_vals)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Compute TD target and error
        next_q = max(Q(u, next_state, a) for a in range(n_actions))
        target = reward + gamma * next_q
        current = Q(v, state, action)
        delta = target - current

        phi_vec = phi(state, action)
        u += alpha * (phi_vec * current - u)
        v += beta * phi_vec * delta

        state = next_state
        if done:
            break

    writer.add_scalar("Reward", total_reward, episode)

env.close()
writer.close()
