from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from itertools import cycle
import gym3
import gym
from gym.spaces import Box, Discrete
from gym3.env import Env
from gym3.internal import misc
from gym3.types_np import concat, split
from gym3 import types_np
from gym3.concat import ConcatEnv
from gym3.env import Env
from gym3.internal import misc
from gym3.subproc import SubprocEnv
from gym3.interop import _make_gym_env


"""
multi task wrapper needs to make multiple different envs
compatible with the same action and observations space

"""

def get_dim(space):

    if type(space) == Box:
        return space.shape[0]

    elif type(space) == Discrete:
        return space.n


def get_dims(env_names):
    ac = []
    ob = []
    rew = []

    for name in env_names:
        print(name)
        o, a, r = get_env_dim(name)
        ob.append(o)
        ac.append(a)
        rew.append(r)

    return np.max(ob), np.max(ac), np.max(rew)


def get_env_dim(env_name):

    env = gym.make(env_name)
    o = get_dim(env.observation_space)
    a = get_dim(env.action_space)
    r1 = abs(env.reward_range[0])
    r2 = abs(env.reward_range[1])


    return o, a, np.max([r1, r2])

def make_multi_task(env_name, ob, ac, seed=None):
    return MultiTaskWrapper(env_name, ob, ac, seed=seed)


def vectorize_multi_task(
    env_names,
    num,
    ob,
    ac,
    seed=None

    ):

    envs = []

    env_names = cycle(env_names)

    for env in env_names:

        envs.append(
        SubprocEnv(
            env_fn=_make_gym_env,
            env_kwargs={
                'env_kwargs':{"env_name":env, "ob":ob, "ac":ac, "seed":seed},
                'env_fn':make_multi_task
                },
            )
        )


        if len(envs) >= num:
            break

    return ConcatEnv(envs)


class MultiTaskWrapper(object):


        def __init__(self, env_name, ob_space, ac_space, seed=None):

                self.env = gym.make(env_name)

                self.ac_space = ac_space
                self.ob_space = ob_space

                self.observation_space = Box(0, ob_space, shape=[ob_space])
                self.action_space = Discrete(ac_space)

                self.real_ac = self.get_dim(self.env.action_space)
                #self.ob_space = self.get_dim(self.env.observation_space)

                env_scales = {
                    "Acrobot-v1":1,
                    "MountainCar-v0":200,
                    "CartPole-v0":200,
                    "LunarLander-v2":400,
                    }

                self.r_scale = env_scales[env_name]

                if seed != None:
                    self.seed(seed)

        def seed(self, seed):
            self.env.seed(seed)

        def scale_reward(self, r):
            pass

        def get_dim(self, space):

                if type(space) == Box:
                        return space.shape[0]

                elif type(space) == Discrete:
                        return space.n


        def step(self, action):

                if action >= self.real_ac:
                        return self.state, -10, False, None

                else:
                        self.state, self.reward, self.done, _ = self.env.step(action)

                        self.convert_state()

                #/self.r_scale
                return self.state, self.reward/self.r_scale, self.done, None

        def convert_state(self):
            state = np.zeros([self.ob_space])
            state[0:self.state.shape[0]] = self.state
            self.state = state


        def reset(self):
                self.state = self.env.reset()
                self.convert_state()
                return self.state


def main():

    """
    #env = MultiTaskWrapper("LunarLander-v2", 8, 8)
    #env = MultiTaskWrapper("CartPole-v0", 8, 8)
    #env = MultiTaskWrapper("MountainCar-v0", 8, 8)
    #env = MultiTaskWrapper("Pendulum-v0", 8, 8)
    env_names = [
            "Pendulum-v0",
            "MountainCar-v0",
            "CartPole-v0",
            "LunarLander-v2"
            ]

    #env = MultiTaskWrapper("LunarLander-v2", 8, 4, 10)
    #env.reset()
    #print(env.step(5))

    env = gym3.vectorize_gym(
            num=4,
            env_fn=make_multi_task,
            env_kwargs={
            "env_name":"CartPole-v0",
            "ob":8,
            "ac":8,
            "r_scale":1,
            }
            )
    """

    env_names = ["Acrobot-v1","MountainCar-v0", "CartPole-v0", "LunarLander-v2"]

    env = vectorize_multi_task(env_names, num=4, ob=8, ac=4, seed=0)



if __name__ == "__main__":
    main()
