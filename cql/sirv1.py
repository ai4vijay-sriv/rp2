import gymnasium as gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Environment
env = gym.make("MountainCar-v0")
n_actions = env.action_space.n
obs_dim = env.observation_space.shape[0]  # Should be 2 (position, velocity)


log_name = f"mountaincar_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
writer = SummaryWriter(log_dir=f"runs/lfa/{log_name}")

# Hyperparameters
alpha = 0.01
gamma = 0.99
epsilon_start = 0.5
epsilon_final = 0.05
exploration_fraction = 0.1
num_episodes = 50000

# Feature vector: just the state (position, velocity)
def phi(state):
    return np.array(state, dtype=np.float32)

# Linear weights: (n_actions x obs_dim)
weights = np.zeros((n_actions, obs_dim), dtype=np.float32)

# ε-greedy schedule
def get_epsilon(episode):
    progress = min(episode / (exploration_fraction * num_episodes), 1.0)
    return epsilon_start + (epsilon_final - epsilon_start) * progress

# Q-value: dot product between action weight and state
def q_value(state, action):
    return np.dot(weights[action], phi(state))

# Policy: ε-greedy
def select_action(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    q_vals = [q_value(state, a) for a in range(n_actions)]
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
        q_next = max(q_value(next_state, a) for a in range(n_actions))
        td_target = reward + gamma * q_next
        td_error = td_target - q_value(state, action)

        # Gradient update
        weights[action] += alpha * td_error * phi(state)

        state = next_state

    # Log to TensorBoard
    writer.add_scalar("Reward", total_reward, episode)
    writer.add_scalar("Epsilon", epsilon, episode)
    print(f"Episode {episode+1}/{num_episodes} | Total reward: {total_reward:.1f}")


env.close()
writer.close()
