# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys
import random

class KingWindyGridWorldEnv(gym.Env):
    '''Creates the King Windy GridWorld Environment'''
    def __init__(self, GRID_HEIGHT=50, GRID_WIDTH=50,
                 WIND = [2, 0, 1, 2, 1, 0, 0, 1, 2, 2, 0, 1, 1, 2, 0, 0, 1, 1, 2, 0, 2, 1, 0, 1, 2, 0, 2, 1, 0, 2, 2, 1, 1, 0, 2, 1, 0, 0, 2, 1, 2, 0, 1, 1, 2, 0, 2, 1, 0, 1],
                 START_STATE = (3, 0), GOAL_STATE = (37, 49),
                 REWARD = -1):
        self.grid_height = GRID_HEIGHT
        self.grid_width = GRID_WIDTH
        self.wind = WIND
        self.start_state = START_STATE
        self.goal_state = GOAL_STATE
        self.observation = START_STATE
        self.reward = REWARD
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(
            low=np.array([0, 0], dtype=np.float32),
            high=np.array([self.grid_height - 1, self.grid_width - 1], dtype=np.float32),
            dtype=np.float32
        )
        '''self.observation_space = spaces.Tuple((
                spaces.Discrete(self.grid_height),
                spaces.Discrete(self.grid_width)))'''
        self.actions = { 'U':0,   #up
                         'R':1,   #right
                         'D':2,   #down
                         'L':3,   #left
                         'UR':4,  #up-right
                         'DR':5,  #down-right
                         'DL':6,  #down-left
                         'UL':7 } #up-left

        # set up destinations for each action in each state
        self.action_destination = np.empty((self.grid_height,self.grid_width), dtype=dict)
        for i in range(0, self.grid_height):
            for j in range(0, self.grid_width):
                destination = dict()
                destination[self.actions['U']] = (max(i - 1 - self.wind[j], 0), j)
                destination[self.actions['D']] = (max(min(i + 1 - self.wind[j], self.grid_height - 1), 0), j)
                destination[self.actions['L']] = (max(i - self.wind[j], 0), max(j - 1, 0))
                destination[self.actions['R']] = (max(i - self.wind[j], 0), min(j + 1, self.grid_width - 1))
                destination[self.actions['UR']] = (max(i - 1 - self.wind[j], 0), min(j + 1, self.grid_width - 1))
                destination[self.actions['DR']] = (max(min(i + 1 - self.wind[j], self.grid_height - 1), 0), min(j + 1, self.grid_width - 1))
                destination[self.actions['DL']] = (max(min(i + 1 - self.wind[j], self.grid_height - 1), 0), max(j - 1, 0))         
                destination[self.actions['UL']] = (max(i - 1 - self.wind[j], 0), max(j - 1, 0))
                self.action_destination[i,j] = destination

    def step(self, action):
        """
        Parameters
        ----------
        action : 0 = Up, 1 = Right, 2 = Down, 3 = Left, 4 = Up-right, 
                 5 = Down-right, 6 = Down-left, 7 = Up-left
        Returns
        -------
        ob, reward, terminated, truncated, info : tuple
        """
        assert self.action_space.contains(action)
        self.observation = self.action_destination[self.observation][action]
        terminated = self.observation == self.goal_state
        truncated = False
        obs_array = np.array(self.observation, dtype=np.float32)
        return obs_array, -1, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        ''' resets the agent position back to the starting position'''
        super().reset(seed=seed)
        self.observation = (self.np_random.integers(0, 50), self.np_random.integers(0, 50))
        obs_array = np.array(self.observation, dtype=np.float32)
        return obs_array, {}

    def render(self, mode='human'):
        ''' Renders the environment. Code borrowed and then modified 
            from
            https://github.com/openai/gym/blob/master/gym/envs/toy_text/cliffwalking.py'''
        outfile = sys.stdout
        shape = (self.grid_height, self.grid_width)

        outboard = ""
        for y in range(-1, self.grid_height + 1):
            outline = ""
            for x in range(-1, self.grid_width + 1):
                position = (y, x)
                if self.observation == position:
                    output = "X"
                elif position == self.goal_state:
                    output = "G"
                elif position == self.start_state:
                    output = "S"
                elif x in {-1, self.grid_width } or y in {-1, self.grid_height}:
                    output = "#"
                else:
                    output = " "

                if position[1] == shape[1]:
                    output += '\n'
                outline += output
            outboard += outline
        outboard += '\n'
        outfile.write(outboard)

    def seed(self, seed=None):
        pass

