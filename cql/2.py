import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime



log_name = f"mountaincar_rbf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

writer = SummaryWriter(log_dir=f"runs/lfa/{log_name}")

# Create environment
env = gym.make("MountainCar-v0")
n_actions = env.action_space.n

# Feature grid resolution (l x l)
l = 2  # Use l = 2 or 3 as per paper guidance

# RBF centers in position and velocity space
pos_min, pos_max = env.observation_space.low[0], env.observation_space.high[0]
vel_min, vel_max = env.observation_space.low[1], env.observation_space.high[1]

pos_centers = np.linspace(pos_min, pos_max, l)
vel_centers = np.linspace(vel_min, vel_max, l)
rbf_centers = np.array([(p, v) for p in pos_centers for v in vel_centers])  # l^2 centers

# Standard deviation for RBFs
sigma_pos = (pos_max - pos_min) / (l - 1) / 1.8
sigma_vel = (vel_max - vel_min) / (l - 1) / 1.8

def rbf_features(state):
    pos, vel = state
    diffs = rbf_centers - np.array([pos, vel])
    feats = np.exp(-((diffs[:, 0] / sigma_pos) ** 2 + (diffs[:, 1] / sigma_vel) ** 2))
    return feats

# Final feature vector φ(s, a) = [rbfs... , one_hot(action)]
def phi(state, action):
    rbf = rbf_features(state)
    rbf = rbf / np.linalg.norm(rbf)  # normalize ‖φ(s)‖ = 1
    a_onehot = np.zeros(n_actions)
    a_onehot[action] = 1
    return np.concatenate([rbf, a_onehot])  # total dim: l^2 + n_actions

# Setup
n_feat = len(phi(env.reset()[0], 0))  # l^2 + 3
theta = np.zeros(n_feat)

alpha = 0.5
gamma = 0.99
n_episodes = 50000

# Epsilon decay
epsilon_start = 0.5
epsilon_min = 0.05
epsilon_decay = 0.995

def q_value(state, action):
    return np.dot(theta, phi(state, action))

def epsilon_greedy(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    return np.argmax([q_value(state, a) for a in range(n_actions)])

rewards = []

for ep in range(n_episodes):
    epsilon = max(epsilon_min, epsilon_start * (epsilon_decay ** ep))
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
    print(f"[Ep {ep + 1}] Reward = {total_reward} | Epsilon = {epsilon:.4f}")
    writer.add_scalar("Episode Reward", total_reward, ep)
    writer.add_scalar("Epsilon", epsilon, ep)

env.close()
plt.plot(rewards)
plt.title("MountainCar: Q-learning with Normalized RBF Features")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid()
plt.show()