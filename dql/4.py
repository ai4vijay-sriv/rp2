import numpy as np
import random
import matplotlib.pyplot as plt

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

state_value = 0
state_value1 = [0, 0]  # For Double Q learning
state_value2 = [0, 0, 0]  # For Triple Q learning


q_values = []          # Store Q-learning value over time
dq_values_avg = []     # Store average of Double Q-learning values over time
tq_values_avg = []     # Store average of Triple Q-learning values over time

Qa = [0, 0]
print(" ")
print("Initial Q value in Q learning:", state_value)
print("Initial Q value in Double Q learning:", state_value1)

for step in range(100):
    axon = 0
    print("------")
    print(f"Step {step + 1}")
    print("------")
    print("Q learning update")
    max_nextstate_value, action = next_maxqa(0)
    state_value += 0.1 * (0 + max_nextstate_value - state_value)
    print("chosen action ",action,"at q learning : ", max_nextstate_value)
    print("Updated Q value: ", state_value)
    axon+=action 
    print(" ")

    print("Double Q learning update")
    if np.random.rand() < 0.5:
        max_nextstate_value, action = next_maxqa(0)
        target = 0 + Q_value(0, action)
        state_value1[0] += 0.1 * (target - state_value1[0])
        print("chosen action ",action,"at double q learning(1) : ", target)
        print("Updated Double Q value: ", state_value1[0])
    else:
        max_nextstate_value, action = next_maxqa(0)
        target = 0 + Q_value(0, action)
        state_value1[1] += 0.1 * (target - state_value1[1])
        print("chosen action ",action,"at double q learning(2) : ", target)
        print("Updated Double Q value: ", state_value1[1])
    axon += action
    print(" ")

    print("Triple Q learning update")
    choice = random.choice([1, 2, 3])
    if choice == 1:
        max_nextstate_value, action = next_maxqa(0)
        target = 0 + (Q_value(0, action) + Q_value(0, action)) / 2
        state_value2[0] += 0.1 * (target - state_value2[0])
        print("chosen action ",action,"at triple q learning(1) : ", target)
        print("Updated Triple Q value: ", state_value2[0])
    elif choice == 2:
        max_nextstate_value, action = next_maxqa(0)
        target = 0 + (Q_value(0, action) + Q_value(0, action)) / 2
        state_value2[1] += 0.1 * (target - state_value2[1])
        print("chosen action ",action,"at triple q learning(2) : ", target)
        print("Updated Triple Q value: ", state_value2[1])
    else:
        max_nextstate_value, action = next_maxqa(0)
        target = 0 + (Q_value(0, action) + Q_value(0, action)) / 2
        state_value2[2] += 0.1 * (target - state_value2[2])
        print("chosen action ",action,"at triple q learning(3) : ", target)
        print("Updated Triple Q value: ", state_value2[2])

    if action == 1 and axon == 2:
        print("yaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaay")


    # Save values for plotting
    q_values.append(state_value)
    dq_values_avg.append(np.mean(state_value1))
    tq_values_avg.append(np.mean(state_value2))


print("-------------------------------------------------------")
print("finally Q value in Q learning:", state_value)
print("finally Q values in Double Q learning:", state_value1)
print("finally Q values in Triple Q learning:", state_value2)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(q_values, label="Q-learning")
plt.plot(dq_values_avg, label="Double Q-learning (avg)")
plt.plot(tq_values_avg, label="Triple Q-learning (avg)")
plt.xlabel("Steps")
plt.ylabel("Estimated Value of State 0")
plt.title("Q-learning vs Double Q-learning Value  vs Triple Q-learning")
plt.legend()
plt.grid(True)
plt.show()
