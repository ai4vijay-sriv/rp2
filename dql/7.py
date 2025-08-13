# this is to implement cql on the example that sir gave
import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter  # <-- added

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

def Q_value(state, action):
    if state == 0 and action == 0:
        return 5
    elif state == 0 and action == 1:
        return 10 if random.random() < 0.1 else -1

def q_value_u(state, action):
    return np.dot(weights_u[action], phi(state))

def q_value_v(state, action):
    return np.dot(weights_v[action], phi(state))

def q_value_w1(state, action):
    return np.dot(weights_w1[action], phi(state))

def q_value_w2(state, action):
    return np.dot(weights_w2[action], phi(state))

def phi(state):
    vec = np.zeros(5, dtype=np.float32)
    vec[state] = 1.0
    return vec

def get_epsilon(episode):
    progress = min(episode / (exploration_fraction * num_episodes), 1.0)
    return epsilon_start + (epsilon_final - epsilon_start) * progress

def get_reward(state, action):
    if state == 0:
        return 0
    elif state == 1 and action == 0:
        return 5
    elif state == 1 and action == 1:
        return 10 if random.random() < 0.1 else -1
    else:
        return 0

def step(state, action):
    if state == 0:
        return (1, 0)  # In this simple example, the next state is always the same
    elif state == 1:
        if action == 0:
            return (2, 5)
        elif action == 1:
            return (3, 10) if random.random() < 0.5 else (4, 1)

n_actions = 2
feature_dim = 5
epsilon_start = 0.5
epsilon_final = 0.05
exploration_fraction = 0.5
num_episodes = 200
gamma = 1
alpha = 0.0001
beta = 0.01

weights_u = np.zeros((n_actions, feature_dim), dtype=np.float32)
weights_v = np.zeros((n_actions, feature_dim), dtype=np.float32)
weights_w1 = np.zeros((n_actions, feature_dim), dtype=np.float32)
weights_w2 = np.zeros((n_actions, feature_dim), dtype=np.float32)

# ===== TensorBoard: init writer =====
run_name = f"cql/double/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
writer = SummaryWriter(log_dir=f"runs/{run_name}")

returns = []
global_step = 0  # per-step counter for TensorBoard

for episode in range(num_episodes):
    state = 0
    done = False
    ep_return = 0
    ep_steps = 0
    td_err = None  # will log last TD error of the episode
    eps = get_epsilon(episode)

    while not done:
        if np.random.rand() < eps:
            action = random.choice([0, 1])
        else:
            q_valsv = [q_value_v(state, a) for a in range(n_actions)]
            q_valsw2 = [q_value_w2(state, a) for a in range(n_actions)]
            q_mean = np.mean([q_valsv, q_valsw2], axis=0)
            action = int(np.argmax(q_mean))

        next_state, reward = step(state, action)
        ep_return += reward
        if next_state != 1:
            done = True


        next_action1 = np.argmax([q_value_w1(next_state, a) for a in range(n_actions)])
        q_next1 = q_value_u(next_state, next_action1)
        td_target1 = reward + gamma * q_next1
        td_error1 = td_target1 - q_value_v(state, action)

        weights_u[action] += alpha * ((phi(state) * q_value_v(state, action)) - weights_u[action])
        weights_v[action] += beta * td_error1 * phi(state)

        td_err = float(td_error1)

        next_action2 = np.argmax([q_value_u(next_state, a) for a in range(n_actions)])
        q_next2 = q_value_w1(next_state, next_action2)
        td_target2 = reward + gamma * q_next2
        td_error2 = td_target2 - q_value_w2(state, action)

        weights_w1[action] += alpha * ((phi(state) * q_value_w2(state, action)) - weights_w1[action])
        weights_w2[action] += beta * td_error2 * phi(state)

        td_err = float(td_error2)

        # clip to keep things stable
        weights_u = np.clip(weights_u, -1e6, 1e6)
        weights_v = np.clip(weights_v, -1e6, 1e6)
        weights_w1 = np.clip(weights_w1, -1e6, 1e6)
        weights_w2 = np.clip(weights_w2, -1e6, 1e6)

        # ===== TensorBoard: per-step logs =====
        writer.add_scalar("train/td_error", td_err, global_step)
        # log norms to catch divergence/exploding weights
        writer.add_scalar("weights/||u||_F", np.linalg.norm(weights_u), global_step)
        writer.add_scalar("weights/||v||_F", np.linalg.norm(weights_v), global_step)
        writer.add_scalar("weights/||w1||_F", np.linalg.norm(weights_w1), global_step)
        writer.add_scalar("weights/||w2||_F", np.linalg.norm(weights_w2), global_step)

        state = next_state
        ep_steps += 1
        global_step += 1

    returns.append(ep_return)

    # ===== TensorBoard: per-episode logs =====
    writer.add_scalar("episode/return", ep_return, episode)
    writer.add_scalar("episode/epsilon", eps, episode)
    writer.add_scalar("episode/steps", ep_steps, episode)

    # (Optional) histograms every few episodes
    if (episode + 1) % 10 == 0:
        writer.add_histogram("hist/weights_u", weights_u, episode)
        writer.add_histogram("hist/weights_v", weights_v, episode)
        writer.add_histogram("hist/weights_w1", weights_w1, episode)
        writer.add_histogram("hist/weights_w2", weights_w2, episode)

    print(f"Episode {episode + 1}: Return = {ep_return}")

print(np.mean(returns))
writer.add_scalar("summary/avg_return", float(np.mean(returns)), num_episodes - 1)

# close the writer cleanly
writer.close()

