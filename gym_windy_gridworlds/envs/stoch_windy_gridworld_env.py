# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import sys
from io import StringIO  # Added missing import

class StochWindyGridWorldEnv(gym.Env):
    '''Creates the Stochastic Windy GridWorld Environment
       NOISE_CASE = 1: the noise is a scalar added to the wind tiles, i.e,
                       all wind tiles are changed by the same amount              
       NOISE_CASE = 2: the noise is a vector added to the wind tiles, i.e,
                       wind tiles are changed by different amounts.
    '''
    def __init__(self, GRID_HEIGHT=7, GRID_WIDTH=10,
                 WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0],
                 START_CELL = (3, 0), GOAL_CELL = (3, 7),
                 REWARD = -1, RANGE_RANDOM_WIND=2,
                 PROB=[0.35, 0.1, 0.1, 0.1, 0.35],
                 NOISE_CASE = 1,
                 SIMULATOR_SEED = 3323,
                 GAMMA = 0.9):
        self.prng_simulator = np.random.RandomState(SIMULATOR_SEED)
        self.grid_height = GRID_HEIGHT
        self.grid_width = GRID_WIDTH
        self.grid_dimensions = (self.grid_height, self.grid_width)
        self.wind = np.array(WIND)
        self.realized_wind = self.wind
        self.start_cell = START_CELL
        self.goal_cell = GOAL_CELL
        self.start_state = self.dim2to1(START_CELL)
        self.goal_state = self.dim2to1(GOAL_CELL)
        self.reward = REWARD
        self.range_random_wind = RANGE_RANDOM_WIND
        self.w_range = np.arange(-self.range_random_wind, self.range_random_wind + 1)
        self.probabilities = PROB
        self.w_prob = dict(zip(self.w_range, self.probabilities))
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple((
                spaces.Discrete(self.grid_height),
                spaces.Discrete(self.grid_width)))
        self.seed()
        self.actions = { 'U':0, 'R':1, 'D':2, 'L':3 }
        self.nA = len(self.actions)
        self.nS = self.dim2to1((self.grid_height-1,self.grid_width-1)) + 1
        self.num_wind_tiles = np.count_nonzero(self.wind)
        self.noise_case = NOISE_CASE
        self.P = []
        self.all_possible_wind_values = np.unique(self.w_range[:, None] + self.wind)
        self.f = np.zeros((self.nS, self.nA, len(self.w_range)), dtype=int)
        for s in range(self.nS):
            for w in (self.w_range + self.range_random_wind):
                if s == self.goal_state:
                    for a in self.actions.values():
                        self.f[s, a, w] = self.goal_state
                else:
                    i, j = self.dim1to2(s)
                    wind = self.wind[j] + w - self.range_random_wind if self.wind[j] != 0 else 0
                    self.f[s, self.actions['U'], w] = self.dim2to1((max(i - 1 - wind, 0), j))
                    self.f[s, self.actions['R'], w] = self.dim2to1((min(max(i - wind, 0), self.grid_height - 1), min(j + 1, self.grid_width - 1)))
                    self.f[s, self.actions['D'], w] = self.dim2to1((max(min(i + 1 - wind, self.grid_height - 1), 0), j))
                    self.f[s, self.actions['L'], w] = self.dim2to1((min(max(i - wind, 0), self.grid_height - 1), max(j - 1, 0)))
        self.P = np.zeros((self.nS, self.nA, self.nS))
        for s in range(self.nS):
            for a in range(self.nA):
                unique_next = np.unique(self.f[s, a, :])
                self.P[s, a][unique_next] = [list(self.f[s, a, :]).count(v) / float(len(self.f[s, a, :])) for v in unique_next]
        self.gamma = GAMMA
        self.trans = np.empty((self.nS, self.nA), dtype=object)
        for s in range(self.nS):
            for a in range(self.nA):
                self.trans[s, a] = np.vstack((
                    self.P[s, a, np.where(self.P[s, a] != 0)][0],
                    np.where(self.P[s, a] != 0)[0],
                    self.reward_func(s, a),
                    np.where(self.P[s, a] != 0)[0] == self.goal_state
                )).T

    def reward_func(self, state, action):
        next_states = np.unique(self.f[state, action, :])
        reward = np.ones(len(next_states)) * -1
        reward[np.where(next_states == self.goal_state)] = 0
        return reward

    def create_absorption_MDP_P(self):
        self.P_new = np.zeros((self.nS + 1, self.nA, self.nS + 1))
        self.P_new[:, :, self.nS] = 1 - self.gamma
        self.P_new[0:self.nS, :, 0:self.nS] = self.gamma * self.P
        self.P_new[self.nS, :, self.nS] = 1

    def _virtual_step_absorb(self, s, a, force_noise=None):
        noise = force_noise if force_noise is not None else self.np_random.choice(self.w_range, 1, p=self.probabilities)[0]
        wind = np.copy(self.wind)
        wind[np.where(wind > 0)] += noise
        if s == self.nS:
            return self.nS, 0, True, wind, noise
        P = np.zeros((self.nS + 1, self.nA, self.nS + 1))
        newS = self.f[s, a, noise + self.range_random_wind]
        P[s, a, newS] = self.gamma
        P[s, a, self.nS] = 1.0 - self.gamma
        prob = P[s, a, np.nonzero(P[s, a, :])][0].tolist()
        destination = self.np_random.choice(np.append(newS, self.nS), 1, p=prob)[0]
        if destination == self.goal_state:
            reward = 0
            isdone = False
        elif destination == self.nS:
            reward = 0
            isdone = True
        else:
            reward = -1
            isdone = False
        return destination, reward, isdone, wind, noise

    def simulate_sample_path(self):
        tau = self.prng_simulator.geometric(p=1 - self.gamma, size=1)[0]
        sample_path = self.prng_simulator.choice(self.w_range, tau, p=self.probabilities)
        return sample_path

    def step_absorb(self, action, force_noise=None):
        assert self.action_space.contains(action)
        self.observation, reward, isdone, wind, noise = self._virtual_step_absorb(self.observation, action, force_noise)
        self.realized_wind = wind
        terminated = isdone
        truncated = False
        return self.observation, reward, terminated, truncated, {'noise': noise}

    def dim2to1(self, cell):
        return np.ravel_multi_index(cell, self.grid_dimensions)

    def dim1to2(self, state):
        return np.unravel_index(state, self.grid_dimensions)

    def _virtual_step_f(self, state, action, force_noise=None):
        noise = self.np_random.choice(self.w_range, 1, p=self.probabilities)[0] if force_noise is None else force_noise
        wind = np.copy(self.wind)
        wind[np.where(wind > 0)] += noise
        destination = self.f[state, action, noise + self.range_random_wind]
        if state == self.goal_state and destination == self.goal_state:
            reward = 0
            isdone = True
        elif state != self.goal_state and destination == self.goal_state:
            reward = 0
            isdone = True
        else:
            reward = -1
            isdone = False
        return destination, reward, isdone, wind, noise

    def step(self, action, force_noise=None):
        assert self.action_space.contains(action)
        self.observation, reward, isdone, wind, noise = self._virtual_step_f(self.observation, action, force_noise)
        self.realized_wind = wind
        terminated = isdone
        truncated = False
        return self.observation, reward, terminated, truncated, {'noise': noise}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.observation = self.start_state
        self.realized_wind = self.wind
        return self.observation, {}

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        for s in range(self.nS):
            position = self.dim1to2(s)
            if self.observation == s:
                output = " x "
            elif position == self.goal_cell:
                output = " T "
            else:
                output = " o "
            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.grid_dimensions[1] - 1:
                output = output.rstrip()
                output += "\n"
            outfile.write(output)
        for i in range(len(self.realized_wind)):
            output = ' ' + str(self.realized_wind[i]) + ' '
            if i == 0:
                output = output.lstrip()
            if i == len(self.realized_wind) - 1:
                output = output.rstrip()
                output += "\n"
            outfile.write(output)
        outfile.write("\n")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

