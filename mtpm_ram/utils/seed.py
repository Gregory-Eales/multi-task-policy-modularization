import numpy as np
import random
import torch
import gym


def set_seed(env, seed):
	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
	env.seed(seed)