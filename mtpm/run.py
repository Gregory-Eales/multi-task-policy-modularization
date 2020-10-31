import os
import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from procgen import ProcgenEnv
import gym3
from procgen import ProcgenGym3Env
import numpy as np
import gym
from modules import *

#from ppo.ppo import PPO


def run(params):

    env = gym.make(params.env_name)

    agent = PPO(params)
    agent.actor.load_state_dict(torch.load("./experiments/Acrobot-Multi-Task-Baseline/params/Acrobot-v1_model.pt"))

    done = False

    state = env.reset()

    rewards = []

    while not done:

        action = agent.act_det(state)

        state, reward, done, info = env.step(action)

        env.render()

        rewards.append(reward)


    print(np.sum(rewards))

if __name__ == '__main__':

    parser = ArgumentParser(add_help=True)

    parser.add_argument('--env_name', default="Acrobot-v1", type=str)
    parser.add_argument('--batch_sz', default=256, type=int)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--k_epochs', default=4, type=int)
    parser.add_argument('--actor_lr', default=5e-4, type=float)
    parser.add_argument('--critic_lr', default=5e-4, type=float)
    parser.add_argument('--epsilon', default=0.2, type=float)
    parser.add_argument('--vision', default=False, type=bool)

    params = parser.parse_args()

    run(params)


