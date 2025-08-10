import gymnasium as gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime, date
from dataclasses import dataclass
import tyro
from pprint import pformat


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
    num_episodes: int = 1000
    l: int = 2
    log_base_dir: str = "runs/lfa"
    today: str = date.today()
    eta = 0.5
    num_runs: int = 10
    eval_episodes: int = 10000


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
pos_grid, vel_grid = np.meshgrid(pos_centers, vel_centers)
rbf_centers = np.column_stack([pos_grid.ravel(), vel_grid.ravel()])

# RBF bandwidth (σ)
sigma_pos = args.eta * 1.8
sigma_vel = args.eta * 0.14

# Feature vector: RBFs over state
def phi(state):
    state = np.array(state)
    diffs = rbf_centers - state
    exponents = -((diffs[:, 0] ** 2) / (2 * sigma_pos ** 2) + (diffs[:, 1] ** 2) / (2 * sigma_vel ** 2))
    return np.exp(exponents).astype(np.float32)

feature_dim = len(phi(env.reset()[0]))

# Q-value: dot product between action weight and RBF feature vector
def q_value_u(state, action, weights_u):
    return np.dot(weights_u[action], phi(state))

def q_value_v(state, action, weights_v):
    return np.dot(weights_v[action], phi(state))

# Policy: ε-greedy
def select_action(state, epsilon, weights_v):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    q_vals = [q_value_v(state, a, weights_v) for a in range(n_actions)]
    return int(np.argmax(q_vals))

# Evaluation: greedy policy only
def evaluate_greedy_policy(env, weights_v, num_episodes):
    rewards = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            q_vals = [q_value_v(state, a, weights_v) for a in range(n_actions)]
            action = int(np.argmax(q_vals))
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state
        rewards.append(total_reward)
    return np.mean(rewards)

# Main loop for multiple training runs
final_eval_rewards = []

for run in range(args.num_runs):
    print(f"\n=== RUN {run+1} / {args.num_runs} ===")
    weights_u = np.zeros((n_actions, feature_dim), dtype=np.float32)
    weights_v = np.zeros((n_actions, feature_dim), dtype=np.float32)

    for episode in range(args.num_episodes):
        state, _ = env.reset()
        done = False
        epsilon = args.epsilon_start + (args.epsilon_final - args.epsilon_start) * \
                  min(episode / (args.exploration_fraction * args.num_episodes), 1.0)

        total_reward = 0
        while not done:
            action = select_action(state, epsilon, weights_v)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            q_next = max(q_value_u(next_state, a, weights_u) for a in range(n_actions))
            td_target = reward + args.gamma * q_next
            td_error = td_target - q_value_v(state, action, weights_v)

            weights_u[action] += args.alpha * ((phi(state) * q_value_v(state, action, weights_v)) - weights_u[action])
            weights_v[action] += args.beta * td_error * phi(state)

            state = next_state

        # Log training stats (optional)
        if run == 0:
            writer.add_scalar("Reward", total_reward, episode)
            writer.add_scalar("Epsilon", epsilon, episode)

    avg_eval_reward = evaluate_greedy_policy(env, weights_v, args.eval_episodes)
    final_eval_rewards.append(avg_eval_reward)
    print(f"Run {run+1} - Greedy Policy Eval Avg Reward (over {args.eval_episodes} eps): {avg_eval_reward:.2f}")
    writer.add_scalar("Eval_Avg_Reward", avg_eval_reward, run)

# Final average
final_mean = np.mean(final_eval_rewards)
print(f"\n=== FINAL AVERAGE OVER {args.num_runs} RUNS: {final_mean:.2f} ===")
writer.add_scalar("Final_Average_Over_10_Runs", final_mean, 0)

env.close()
writer.close()
