from matplotlib import pyplot as plt
from argparse import ArgumentParser
from procgen import ProcgenGym3Env
from tqdm import tqdm
import numpy as np
import random
import gym3
import time
import copy
import os

from modules import *
from utils import *

"""
########################################
TO DO:

    - how to train routing network? (distance clustering?)
    - classify task using clustering?
    - ways in which gradients can be modified?
    - add experiment folder already exists warning
    - add testing loop with non stochastic action
    - add env wrapper for multitask learning
        - train a model for each vector env with multitask wrapper

    1. train cluster algorithm on latent space
    2. train agent using clustered feature modularizer

########################################
"""

if __name__ == '__main__':

    env_names = [
            "Acrobot-v1",
            "MountainCar-v0",
            "CartPole-v0",
            "LunarLander-v2"
            ]

    parser = ArgumentParser(add_help=True)

    # experiment and  environment
    parser.add_argument(
        '--experiment_name',
        default="CartPole-Single-Task-Baseline",
        type=str
    )

    parser.add_argument('--env_names', default=["CartPole-v0"])

    # saving options
    parser.add_argument('--log', default=True, type=bool)
    parser.add_argument('--graph', default=True, type=bool)

    # training params
    parser.add_argument('--is_multi_task', default=False, type=bool)
    parser.add_argument('--is_procgen', default=False, type=bool)
    parser.add_argument('--random_seeds', default=list(range(5)), type=list)
    parser.add_argument('--n_steps', default=30000, type=int)
    parser.add_argument('--batch_sz', default=64, type=int)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--k_epochs', default=4, type=int)
    parser.add_argument('--n_envs', default=4, type=int)
    parser.add_argument('--update_step', default=1200, type=int)

    # model params
    parser.add_argument('--vision', default=False, type=bool)
    parser.add_argument('--actor_lr', default=5e-3, type=float)
    parser.add_argument('--critic_lr', default=5e-3, type=float)
    parser.add_argument('--epsilon', default=0.2, type=float)
    parser.add_argument('--in_dim', default=4, type=int)
    parser.add_argument('--out_dim', default=2, type=int)
    parser.add_argument('--h_dim', default=64, type=int)

    params = parser.parse_args()

    run_experiment(PPO, params)
