import numpy as np
import random
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

    
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
            return (3,10) if random.random() < 0.1 else (4,1)
    
n_actions = 2
epsilon_start = 0.5
epsilon_final = 0.01
exploration_fraction = 0.5
num_episodes = 20
gamma = 1

q=[[0,0],[0,0],[0,0],[0,0],[0,0]]

q1= [[0,0],[0,0],[0,0],[0,0],[0,0]]  
q2= [[0,0],[0,0],[0,0],[0,0],[0,0]]  

def q_value(state, action):
    return q[state][action]

def q1_value(state, action):
    return q1[state][action]

def q2_value(state, action):
    return q2[state][action]

returns = []  # Store returns for each episode
double_returns = []  # Store returns for Double Q-learning
toggles = []  # Store toggles for Double Q-learning

print("Q-learning")
for episode in range(num_episodes):
    state = 0
    done = False
    ep_return = 0
    exp_action=0
    while not done:
        if np.random.rand() < get_epsilon(episode):
            action = random.choice([0, 1])
            exp_action = 1
        else:
            q_valsv = [q_value(state, a) for a in range(n_actions)]
            action = int(np.argmax(q_valsv))
        
        next_state, reward = step(state, action)
        ep_return += reward
        if next_state != 1:
            done = True
            
        next_action = np.argmax([q_value(next_state, a) for a in range(n_actions)])
        q_next = q_value(next_state, next_action)
        td_target = reward + gamma * q_next
        td_error = td_target - q_value(state, action)  
        q[state][action] += 0.1 * (td_error)  
        if exp_action == 1:
            print(f"Episode {episode + 1}, Step: {state} -> {next_state} with exp Action: {action}, Reward: {reward}")
            exp_action = 0
        else:
            print(f"Episode {episode + 1}, Step: {state} -> {next_state} with Action: {action}, Reward: {reward}")
        print(f"q values: {q}")
        state = next_state
    returns.append(ep_return)

    #print(f"Episode {episode + 1}: Return = {ep_return}")
    #print(q)

print("Double Q-learning")

for episode in range(num_episodes):
    state = 0
    done = False
    ep_return = 0
    exp_action = 0
    while not done:
        if np.random.rand() < get_epsilon(episode):
            action = random.choice([0, 1])
            exp_action = 1
        else:     
            q_valsa = [q1_value(state, a) for a in range(n_actions)]
            q_valsb = [q2_value(state, a) for a in range(n_actions)]
            pairwise_sum = [a + b for a, b in zip(q_valsa, q_valsb)]
            action = int(np.argmax(pairwise_sum))
        
        next_state, reward = step(state, action)
        ep_return += reward
        if next_state != 1:
            done = True
             
        next_action1 = np.argmax([q1_value(next_state, a) for a in range(n_actions)])
        q_next1 = q2_value(next_state, next_action1)
        td_target1 = reward + gamma * q_next1
        td_error1 = td_target1 - q1_value(state, action)
        q1[state][action] += 0.1 * (td_error1)

        next_action2 = np.argmax([q2_value(next_state, a) for a in range(n_actions)])
        q_next2 = q1_value(next_state, next_action2)
        td_target2 = reward + gamma * q_next2
        td_error2 = td_target2 - q2_value(state, action)
        q2[state][action] += 0.1 * (td_error2)

        if exp_action == 1:
            print(f"Episode {episode + 1}, Step: {state} -> {next_state} with exp Action: {action}, Reward: {reward}")
            exp_action = 0
        else:
            print(f"Episode {episode + 1}, Step: {state} -> {next_state} with Action: {action}, Reward: {reward}")
        print(f"q1 values: {q1}")
        print(f"q2 values: {q2}")

        state = next_state
    double_returns.append(ep_return)
    print(" ")

    #print(f"Episode {episode + 1}: Return = {ep_return}")
    #print(q1)
    #print(q2)


#print(q)
#print(q1)
#print(q2)
#print("Toggles:", toggles)
# Plotting
plt.figure(figsize=(10, 6))
plt.plot(returns, label="Q-learning")
plt.plot(double_returns, label="Double Q-learning (avg)")
plt.xlabel("Episode")
plt.ylabel("Return")
plt.title("Q-learning vs Double Q-learning Returns")
plt.legend()
plt.grid(True)
plt.show()


