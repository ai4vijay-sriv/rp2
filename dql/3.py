import numpy as np
import random


def next_maxqa(state):
    if state == 0:
        max_v = np.max([5, 10 if random.random() < 0.1 else -1])
        max_a = 0 if max_v == 5 else 1
        return max_v, max_a

def Q_value(state,action):
    if state == 0 and action == 0:
        return 5
    elif state == 0 and action == 1:
        return 10 if random.random() < 0.1 else -1

state_value = 0
state_value1 = [0, 0]  # For Double Q learning

Qa = [0,0]
print(" ")
print("Initial Q value in Q learning:", state_value) 
print("Initial Q value in Double Q learning:", state_value1)

for step in range(20):
    print("------")
    print(f"Step {step + 1}")
    print("------")
    print("Q learning update")
    max_nextstate_value, action = next_maxqa(0)
    state_value += 0.1 * (0 + max_nextstate_value - state_value)
    
    print("chosen at q learning : ", max_nextstate_value)

    print("Updated Q value: ", state_value)

    print(" ")
    print("Double Q learning update")
    
    if np.random.rand() < 0.5:
        max_nextstate_value, action = next_maxqa(0)
        state_value1[0] += 0.1 *(0 + Q_value(0,action) - state_value1[0])
        print("chosen at double q learning-1 : ", max_nextstate_value)
        print("Updated Double Q value: ", state_value1[0])
    else:
        max_nextstate_value, action = next_maxqa(0)
        state_value1[1] += 0.1 * (0 + Q_value(0,action) - state_value1[1])
        print("chosen at double q learning-2 : ", max_nextstate_value)
        print("Updated Double Q value: ", state_value1[1])

print("-------------------------------------------------------")
print("finally Q value in Q learning:", state_value)
print("finally Q value in Double Q learning:", state_value1)