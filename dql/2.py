import numpy as np
import random

q0 = 0
q1 = 5
q2 = 10
q3 = -1
q4 = 0

dq0 = [0,0]
dq1 = [5,5]
dq2 = [10,10]
dq3 = [-1,-1]
dq4 = [0,0]

def next_maxq(state, action):


for i in range(20):

    print(f"Step {i+1}")
    print("--------------------------------------------------")
    print("Q learning update")
    choice_for_a2 = q2 if random.random() < 0.1 else q3
    print("chosen at q learning", choice_for_a2)
    q0 = q0 + 0.1 * (0 + max(q1, choice_for_a2) - q0)
    #print("Updated Q learning value:", q0)

    print("Double Q learning update")

    if np.random.rand() < 0.5:
        choice_for_a2 = dq2[0] if random.random() < 0.1 else dq3[0]
        dq0[0] = dq0[0] + 0.1 * (0 + choice_for_a2 - dq0[0])
        print("chosen at double q learning 1", choice_for_a2)
        #print("Updated double Q learning value:", dq0[0])

    else:
        choice_for_a2 = dq2[1] if random.random() < 0.1 else dq3[1]
        dq0[1] = dq0[1] + 0.1 * (0 + choice_for_a2 - dq0[1])
        print("chosen at double q learning 2", choice_for_a2)
        #print("Updated double Q learning value:", dq0[1])

    print("Q learning value:", q0)

    print("Double Q learning value:", dq0)
    print("--------------------------------------------------\n")
    


