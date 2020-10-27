from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

import gym
from gym.spaces import Box, Discrete
from gym3.env import Env
from gym3.internal import misc
from gym3.types_np import concat, split




"""
multi task wrapper needs to make multiple different envs
compatible with the same action and observations space


"""

def multi_task_make(env_name):
	return 0

def multi_task_gym(num, env_names, seed):

	"""

	Need to normalize observations

	get max action space
	get max observation space
	get the max and min reward

	give max_min values to each wrapper

	"""

	for _ in range(num):
		pass


class MultiTaskWrapper(object):


	def __init__(self, env_name, ob_space, ac_space, max_reward, min_reward):

		self.env = gym.make(env_name)

		self.ac_space = self.get_dim(self.env.action_space)
		self.ob_space = self.get_dim(self.env.observation_space)

		print( self.ob_space, " -> ", self.ac_space)
	
	
	def get_dim(self, space):

		if type(space) == Box:
			return space.shape[0]

		elif type(space) == Discrete:
			return space.n


	def step(self, action):
		
		if action >= self.ac_space:
			return self.state, -1, True, None

		else:
			self.state, self.reward, self.done, _ = self.env.state(action)
			return self.state, self.reward, self.done

	def reset(self):
		self.state = self.env.reset()
		return self.state


def main():
	
	env = MultiTaskWrapper("LunarLander-v2", 8, 8)
	env = MultiTaskWrapper("CartPole-v0", 8, 8)
	env = MultiTaskWrapper("MountainCar-v0", 8, 8)
	env = MultiTaskWrapper("Pendulum-v0", 8, 8)


	multi_task_gym(
		10,
		["Pendulum-v0", "MountainCar-v0","CartPole-v0", "LunarLander-v2"],
		seed
		)

if __name__ == "__main__":
	main()