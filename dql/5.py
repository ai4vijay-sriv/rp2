#this is to implement cql on the example that sir gave
import numpy as np
import random
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

def next_maxqa(state):
    if state == 0:
        max_v = np.max([5, 10 if random.random() < 0.1 else -1])
        max_a = 0 if max_v == 5 else 1
        return max_v, max_a

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
weights_w1 = np.zeros((n_actions, feature_dim), dtype=np.float32)
weights_w2 = np.zeros((n_actions, feature_dim), dtype=np.float32)

returns = []

for episode in range(num_episodes):
    state = 0
    done = False
    ep_return = 0
    while not done:
        if np.random.rand() < get_epsilon(episode):
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
            
        choice = random.choice([0, 1])
        if choice ==0:
            next_action1 = np.argmax([q_value_w1(next_state, a) for a in range(n_actions)])
            q_next1 = q_value_u(next_state, next_action1)
            td_target1 = reward + gamma * q_next1
            td_error1 = td_target1 - q_value_v(state, action)

            weights_u[action] += alpha * ((phi(state) * q_value_v(state, action)) - weights_u[action])
            weights_v[action] += beta * td_error1 * phi(state)
        else:
            next_action2 = np.argmax([q_value_u(next_state, a) for a in range(n_actions)])
            q_next2 = q_value_w1(next_state, next_action2)
            td_target2 = reward + gamma * q_next2
            td_error2 = td_target2 - q_value_w2(state, action)
            

            weights_w1[action] += alpha * ((phi(state) * q_value_w2(state, action)) - weights_w1[action])
            weights_w2[action] += beta * td_error2 * phi(state)  


        weights_u = np.clip(weights_u, -1e6, 1e6)
        weights_v = np.clip(weights_v, -1e6, 1e6)
        weights_w1 = np.clip(weights_w1, -1e6, 1e6)
        weights_w2 = np.clip(weights_w2, -1e6, 1e6)

        state = next_state
    returns.append(ep_return)

    print(f"Episode {episode + 1}: Return = {ep_return}")

print(np.mean(returns))
plt.plot(returns)
plt.xlabel("Episode")
plt.ylabel("Return")    
plt.title("CQL & DQL Returns Over Episodes")
plt.grid(True)
plt.show()




    

    



