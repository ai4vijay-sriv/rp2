# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys
import random

class WindyGridWorldEnv(gym.Env):
    '''Creates the Windy GridWorld Environment'''
    metadata = {
        "render_modes": ["human"],
        "render_fps": 60
    }

    def __init__(self, GRID_HEIGHT=10, GRID_WIDTH=10,
                 #WIND = [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 2, 2, 3, 2, 1, 1, 1, 1, 0, 0], #for 20 - (3,18)
                 #WIND = [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 2, 2, 1, 1, 1, 0], #for 16 - (3,14)
                 #WIND = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 1, 1, 0], #for 13 -(3,10)
                 WIND = [2,0,1,2,1,0,0,1,2,2],
                 START_STATE=(3, 0), 
                 GOAL_STATE=(3, 7), 
                 REWARD=-1, render_mode=None):
        self.render_mode = render_mode
        self.grid_height = GRID_HEIGHT
        self.grid_width = GRID_WIDTH
        self.wind = WIND
        self.start_state = START_STATE
        self.goal_state = GOAL_STATE
        self.reward = REWARD
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=np.array([0, 0], dtype=np.float32),
            high=np.array([self.grid_height - 1, self.grid_width - 1], dtype=np.float32),
            dtype=np.float32
        )


        self.actions = {'U': 0, 'R': 1, 'D': 2, 'L': 3}

        self.action_destination = np.empty((self.grid_height, self.grid_width), dtype=dict)
        for i in range(0, self.grid_height):
            for j in range(0, self.grid_width):
                destination = dict()
                destination[self.actions['U']] = (max(i - 1 - self.wind[j], 0), j)
                destination[self.actions['D']] = (max(min(i + 1 - self.wind[j], self.grid_height - 1), 0), j)
                destination[self.actions['L']] = (max(i - self.wind[j], 0), max(j - 1, 0))
                destination[self.actions['R']] = (max(i - self.wind[j], 0), min(j + 1, self.grid_width - 1))
                self.action_destination[i, j] = destination
        self.nA = len(self.actions)

    def step(self, action):
        """
        Returns:
        -------
        ob, reward, terminated, truncated, info
        """
        assert self.action_space.contains(action)
        prev_observation = self.observation
        self.observation = self.action_destination[self.observation][action]
        
        terminated = self.observation == self.goal_state
        truncated = False
        #shaped_reward = -1 + 0.1 * (np.linalg.norm(np.array(prev_observation) - np.array(self.goal_state)) - np.linalg.norm(np.array(self.observation) - np.array(self.goal_state)))


        obs_array = np.array(self.observation, dtype=np.float32)
        #return obs_array, shaped_reward, terminated, truncated, {}
        return obs_array, -1, terminated, truncated, {}
        

    def reset(self, seed=None, options=None):
        ''' resets the agent position back to the starting position'''
        super().reset(seed=seed)
        self.observation = self.start_state #(self.np_random.integers(0, 10), self.np_random.integers(0, 10))#self.start_state
        obs_array = np.array(self.observation, dtype=np.float32)
        return obs_array, {}

    def render(self, mode='human'):
        ''' Renders the environment '''
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
                elif x in {-1, self.grid_width} or y in {-1, self.grid_height}:
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

