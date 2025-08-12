import gymnasium as gym
import numpy as np
import gym_windy_gridworlds
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from dataclasses import dataclass
import tyro
import wandb
from pprint import pformat
import random

env = gym.make("WindyGridWorld-v0", render_mode="human")
obs, info = env.reset()

n_actions = env.action_space.n
    
obs_dim = env.observation_space.shape[0]

pos_min, pos_max = env.observation_space.low[0], env.observation_space.high[0]
vel_min, vel_max = env.observation_space.low[1], env.observation_space.high[1]

print(pos_max, pos_min, vel_max, vel_min)