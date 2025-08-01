import numpy as np

# MDP config
n_states = 4
n_actions = 2
gamma = 1.0
alpha = 0.1
np.random.seed(42)

# Terminal values
true_state_values = np.array([0, 5, 10, -1])

# Updated transition function
def transition(state, action):
    if state != 0:
        return state
    if action == 0:
        return 1  # always goes to state 1
    return np.random.choice([2, 3], p=[0.4, 0.6])  # stochastic path

# Reward is always 0 (only future value matters)
def reward(s_prime):
    return 0

# Initialize Q-tables
Q_q = np.zeros((n_states, n_actions))      # Q-learning
Q1 = np.zeros((n_states, n_actions))       # Double Q-learning Q1
Q2 = np.zeros((n_states, n_actions))       # Double Q-learning Q2

# Simulation
n_steps = 500  # You can change this
events = []

for step in range(1, n_steps + 1):
    # Action selection
    q_action = np.argmax(Q_q[0])
    avg_Q = (Q1 + Q2) / 2
    dq_action = np.argmax(avg_Q[0])

    # Track divergence
    if q_action == 1 and dq_action == 0:
        events.append((step, Q_q[0].copy(), avg_Q[0].copy()))

    # Q-learning update
    next_state_q = transition(0, q_action)
    td_target_q = reward(next_state_q) + gamma * np.max(Q_q[next_state_q])
    Q_q[0, q_action] += alpha * (td_target_q - Q_q[0, q_action])

    # Double Q-learning update
    dq_sample_action = np.random.choice([0, 1])
    next_state_dq = transition(0, dq_sample_action)
    r = reward(next_state_dq)
    if np.random.rand() < 0.5:
        a_max = np.argmax(Q1[next_state_dq])
        td_target = r + gamma * Q2[next_state_dq, a_max]
        Q1[0, dq_sample_action] += alpha * (td_target - Q1[0, dq_sample_action])
    else:
        a_max = np.argmax(Q2[next_state_dq])
        td_target = r + gamma * Q1[next_state_dq, a_max]
        Q2[0, dq_sample_action] += alpha * (td_target - Q2[0, dq_sample_action])

# Print overestimation events
print("\n📌 Moments where Q-learning picked action 1 and Double Q-learning preferred action 0:\n")
for step, q_vals, dq_vals in events:
    print(f"Step {step}")
    print(f"  Q-learning Q-values       : {q_vals}")
    print(f"  Double Q-learning Q-values: {dq_vals}")
    print(f"  ⚠️  Overestimation likely\n")

if not events:
    print("❌ No divergence in this run. Increase steps or rerun with new seed.")
else:
    print(f"✅ Found {len(events)} overestimation moment(s) in {n_steps} steps.")
