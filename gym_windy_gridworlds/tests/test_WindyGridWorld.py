# -*- coding: utf-8 -*-
import gymnasium as gym
import gym_windy_gridworlds


env = gym.make('WindyGridWorld-v0')

env.reset()

done = False

while not done:
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    print(observation, reward, done, info)
    print(env.single_observation_space.shape)
    
