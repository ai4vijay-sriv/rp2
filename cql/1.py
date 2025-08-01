import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

log_name = f"mountaincar_rbf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
writer = SummaryWriter(log_dir=f"runs/lfa/{log_name}")

# Environment setup
env = gym.make("MountainCar-v0")
n_actions = env.action_space.n

# RBF setup
l = 2
pos_min, pos_max = env.observation_space.low[0], env.observation_space.high[0]
vel_min, vel_max = env.observation_space.low[1], env.observation_space.high[1]

pos_centers = np.linspace(pos_min, pos_max, l)
vel_centers = np.linspace(vel_min, vel_max, l)
rbf_centers = np.array([(p, v) for p in pos_centers for v in vel_centers])

sigma_pos = (pos_max - pos_min) / (l - 1) / 1.8
sigma_vel = (vel_max - vel_min) / (l - 1) / 1.8

def rbf_features(state):
    pos, vel = state
    diffs = rbf_centers - np.array([pos, vel])
    feats = np.exp(-((diffs[:, 0] / sigma_pos) ** 2 + (diffs[:, 1] / sigma_vel) ** 2))
    return feats

def phi(state, action):
    rbf = rbf_features(state)
    rbf = rbf / np.linalg.norm(rbf)
    a_onehot = np.zeros(n_actions)
    a_onehot[action] = 1
    return np.concatenate([rbf, a_onehot])

# Parameters
n_feat = len(phi(env.reset()[0], 0))
theta = np.zeros(n_feat)

alpha = 0.5
gamma = 0.99
n_episodes = 50000

# Exploration schedule
epsilon_start = 0.5
epsilon_final = 0.05
exploration_fraction = 0.2

def get_epsilon(episode):
    progress = min(episode / (exploration_fraction * n_episodes), 1.0)
    return epsilon_start + (epsilon_final - epsilon_start) * progress

def q_value(state, action):
    return np.dot(theta, phi(state, action))

def epsilon_greedy(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    return np.argmax([q_value(state, a) for a in range(n_actions)])

# Training loop
rewards = []

for ep in range(n_episodes):
    epsilon = get_epsilon(ep)
    state, _ = env.reset()
    total_reward = 0

    for _ in range(200):
        action = epsilon_greedy(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        q_sa = q_value(state, action)
        q_next = max(q_value(next_state, a) for a in range(n_actions))
        target = reward + gamma * q_next if not done else reward

        td_error = target - q_sa
        theta += alpha * td_error * phi(state, action)

        state = next_state
        total_reward += reward
        if done:
            break

    rewards.append(total_reward)
    writer.add_scalar("Reward/Total", total_reward, ep)
    writer.add_scalar("Exploration/Epsilon", epsilon, ep)

    if ep % 100 == 0:
        print(f"[Ep {ep + 1}] Reward = {total_reward} | Epsilon = {epsilon:.4f}")

env.close()
writer.close()

# Optional visualization
plt.plot(rewards)
plt.title("MountainCar: Q-learning with RBF + Linear Epsilon Decay")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid()
plt.show()
