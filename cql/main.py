import gymnasium as gym
import numpy as np
import wandb

def phi(state, action, n_actions):
    pos, vel = state
    pos_norm = (pos + 1.2) / 1.8
    vel_norm = (vel + 0.07) / 0.14
    features = [pos_norm, vel_norm, pos_norm**2, vel_norm**2, pos_norm * vel_norm]
    one_hot = np.zeros(n_actions)
    one_hot[action] = 1.0
    return np.concatenate((features, one_hot))  # shape: (8,)

def train(config=None):
    with wandb.init(config=config, project="convergent_testing"):
        config = wandb.config

        env = gym.make("MountainCar-v0")
        n_actions = env.action_space.n
        w = np.zeros(5 + n_actions)

        epsilon_start = 1.0
        epsilon_end = 0.05
        epsilon_decay = 0.0005
        n_episodes = 500

        for ep in range(n_episodes):
            state = env.reset()[0]
            done = False
            total_reward = 0
            epsilon = max(epsilon_end, epsilon_start * np.exp(-epsilon_decay * ep))

            while not done:
                if np.random.rand() < epsilon:
                    action = np.random.choice(n_actions)
                else:
                    action = np.argmax([np.dot(w, phi(state, a, n_actions)) for a in range(n_actions)])

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                q_sa = np.dot(w, phi(state, action, n_actions))
                max_q_next = max(np.dot(w, phi(next_state, a, n_actions)) for a in range(n_actions))

                td_target = reward + config.gamma * max_q_next
                td_error = td_target - q_sa
                w += config.alpha * td_error * phi(state, action, n_actions)

                total_reward += reward
                state = next_state

            wandb.log({
                "episode": ep,
                "return": total_reward,
                "epsilon": epsilon
            })

        env.close()
