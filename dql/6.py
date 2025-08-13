#this is to implement cql on the example that sir gave
import numpy as np
import random
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
    
def q_value_u(state, action):
    return np.dot(weights_u[action], phi(state))

def q_value_v(state, action):
    return np.dot(weights_v[action], phi(state))

def phi(state):
    vec = np.zeros(5, dtype=np.float32)
    vec[state] = 1.0
    return vec

def get_epsilon(episode):
    progress = min(episode / (exploration_fraction * num_episodes), 1.0)
    return epsilon_start + (epsilon_final - epsilon_start) * progress


def step(state, action):
    if state == 0:
        return (1,0)  # In this simple example, the next state is always the same
    elif state == 1:
        if action == 0:
            return (2,5)
        elif action == 1:
            return (3,10) if random.random() < 0.5 else (4,1)
    

n_actions = 2
feature_dim = 5
epsilon_start = 0.5
epsilon_final = 0.05
exploration_fraction = 0.5
num_episodes = 100
gamma = 1
alpha = 0.01
beta = 0.0001

weights_u = np.zeros((n_actions, feature_dim), dtype=np.float32)
weights_v = np.zeros((n_actions, feature_dim), dtype=np.float32)

returns = []

for episode in range(num_episodes):
    state = 0
    done = False
    ep_return = 0
    while not done:
        if np.random.rand() < get_epsilon(episode):
            action = random.choice([0, 1])
        else:
            action = np.argmax([q_value_v(state, a) for a in range(n_actions)])
        
        next_state, reward = step(state, action)
        ep_return += reward
        if next_state != 1:
            done = True
            

        next_action1 = np.argmax([q_value_u(next_state, a) for a in range(n_actions)])
        q_next1 = q_value_u(next_state, next_action1)
        td_target1 = reward + gamma * q_next1
        td_error1 = td_target1 - q_value_v(state, action)

        weights_u[action] += alpha * ((phi(state) * q_value_v(state, action)) - weights_u[action])
        weights_v[action] += beta * td_error1 * phi(state)
        weights_u = np.clip(weights_u, -1e6, 1e6)
        weights_v = np.clip(weights_v, -1e6, 1e6)

        state = next_state
    returns.append(ep_return)

    print(f"Episode {episode + 1}: Return = {ep_return}")



print(np.mean(returns))
plt.plot(returns)
plt.xlabel("Episode")
plt.ylabel("Return")    
plt.title("CQL Returns Over Episodes")
plt.grid(True)
plt.show()




    

    



