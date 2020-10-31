from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import os
import random
import gym3
from procgen import ProcgenGym3Env
import time
from matplotlib import pyplot as plt
import copy

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
        default="Acrobot-Multi-Task-Baseline",
        type=str
    )

    parser.add_argument('--env_names', default=["Acrobot-v1"])

    # saving options
    parser.add_argument('--log', default=True, type=bool)
    parser.add_argument('--graph', default=True, type=bool)

    # training params
    parser.add_argument('--is_multi_task', default=True, type=bool)
    parser.add_argument('--is_procgen', default=False, type=bool)
    parser.add_argument('--random_seeds', default=list(range(1)), type=list)
    parser.add_argument('--n_steps', default=30000, type=int)
    parser.add_argument('--batch_sz', default=256, type=int)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--k_epochs', default=4, type=int)
    parser.add_argument('--n_envs', default=4, type=int)
    parser.add_argument('--update_step', default=1200, type=int)

    # model params
    parser.add_argument('--vision', default=False, type=bool)
    parser.add_argument('--actor_lr', default=5e-4, type=float)
    parser.add_argument('--critic_lr', default=5e-4, type=float)
    parser.add_argument('--epsilon', default=0.2, type=float)

    params = parser.parse_args()

    run_experiment(PPO, params)
