import numpy as np
import gym
from envs.utils import goal_distance, goal_distance_obs
from utils.os_utils import remove_color
from gym_kuka_mujoco.envs import HammerEnv

class HammerEnvWrapper():
    def __init__(self, args, **env_options):
        self.args = args
        self.env = HammerEnv(**env_options)


