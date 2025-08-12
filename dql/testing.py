import gymnasium as gym
import numpy as np

a=[1,2,3,4]
b=[5,6,7,8]
c = np.add(a, b)
d = np.mean([a, b], axis=0)
print(c)  # Output: [ 6  8 10 12]
print(d)