import numpy as np
import random
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter   # NEW

writer = SummaryWriter()
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
            return (3,10) if random.random() < 0.5 else (4,1)
    
n_actions = 2
epsilon_start = 0.5
epsilon_final = 0.05
exploration_fraction = 0.5
num_episodes = 200
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
    state_d = 0
    done = False
    done_d = False
    ep_return = 0
    ep_return_d = 0
    exp_action=0
    while not done:
        if np.random.rand() < get_epsilon(episode) and episode < num_episodes * 0.1:
            action = random.choice([0, 1])
            action_d = action
            exp_action = 1
        else:
            q_valsv = [q_value(state, a) for a in range(n_actions)]
            action = int(np.argmax(q_valsv))

            q_valsa = [q1_value(state_d, a) for a in range(n_actions)]
            q_valsb = [q2_value(state_d, a) for a in range(n_actions)]
            pairwise_sum = [a + b for a, b in zip(q_valsa, q_valsb)]
            action_d = int(np.argmax(pairwise_sum))
        
        next_state, reward = step(state, action)

        next_state_d, reward_d = step(state_d, action_d)

        ep_return += reward
        ep_return_d += reward_d

        if next_state != 1:
            done = True
        if next_state_d != 1:
            done = True
            
        next_action = np.argmax([q_value(next_state, a) for a in range(n_actions)])
        q_next = q_value(next_state, next_action)
        td_target = reward + gamma * q_next
        td_error = td_target - q_value(state, action)  
        q[state][action] += 0.1 * (td_error)  

        next_action1 = np.argmax([q1_value(next_state_d, a) for a in range(n_actions)])
        q_next1 = q2_value(next_state_d, next_action1)
        td_target1 = reward_d + gamma * q_next1
        td_error1 = td_target1 - q1_value(state_d, action_d)
        q1[state_d][action_d] += 0.1 * (td_error1)

        next_action2 = np.argmax([q2_value(next_state_d, a) for a in range(n_actions)])
        q_next2 = q1_value(next_state_d, next_action2)
        td_target2 = reward_d + gamma * q_next2
        td_error2 = td_target2 - q2_value(state_d, action_d)
        q2[state_d][action_d] += 0.1 * (td_error2)

        if exp_action == 1:
            print(f"Episode {episode + 1}, Step: {state} -> {next_state} with exp Action: {action}, Reward: {reward}")
            print(f"Episode {episode + 1}, Step: {state_d} -> {next_state_d} with exp Action: {action_d}, Reward: {reward_d}")
            exp_action = 0
        else:
            print(f"Episode {episode + 1}, Step: {state} -> {next_state} with Action: {action}, Reward: {reward}")
            print(f"Episode {episode + 1}, Step: {state_d} -> {next_state_d} with Action: {action_d}, Reward: {reward_d}")
        print(f"q values: {q}")
        print(f"q1 values: {q1}")
        print(f"q2 values: {q2}")
        print("returns:", ep_return)
        print("returns_d:", ep_return_d)
        print(" ")
        state = next_state
        state_d = next_state_d
    returns.append(ep_return)
    double_returns.append(ep_return_d)
    #writer.add_scalar("return/q_learning", ep_return, episode)        # NEW
    #writer.add_scalar("return/q_learning", ep_return_d, episode)  # NEW
    writer.add_scalars(
    "return",  # one chart
    {
        "Q-learning": ep_return,
        "Double Q-learning": ep_return_d
    },
    episode
)


# plt.figure(figsize=(10, 6))
# plt.plot(returns, label="Q-learning")
# plt.plot(double_returns, label="Double Q-learning (avg)")
# plt.xlabel("Episode")
# plt.ylabel("Return")
# plt.title("Q-learning vs Double Q-learning Returns")
# plt.legend()
# plt.grid(True)
# plt.show()


