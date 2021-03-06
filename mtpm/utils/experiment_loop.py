import gym
from matplotlib import pyplot as plt
import gym3
from procgen import ProcgenGym3Env


from .dir import *
from .seed import *
from .train_loop import *
from .graph import *
from .data import *

def run_experiment(Agent, hparams):


    # create the path for the experiment
    exp_path = create_exp_dir(hparams.experiment_name)

    if hparams.is_multi_task:

        rewards = []

        for seed in hparams.random_seeds:

            set_seed(seed)

            agent = Agent(hparams)

            if hparams.is_multi_task:
                r = train_multi_task(
                    agent=agent,
                    env_names=hparams.env_names,
                    seed=seed,
                    n_envs=hparams.n_envs,
                    n_steps=hparams.n_steps,
                    update_step=hparams.update_step,
                    )

            rewards.append(r)

        plot_rewards(
            rewards,
            hparams.update_step,
            path=exp_path,
            env_name=hparams.env_names
            )

        save_results(
            env_name=hparams.env_names,
            rewards=rewards,
            path=exp_path,
            update_steps=hparams.update_step,
            seeds=hparams.random_seeds
            )

        save_model(agent, hparams.env_names, exp_path)

    else:
        for env_name in hparams.env_names:
            rewards = []
            for seed in hparams.random_seeds:

                set_seed(seed)

                agent = Agent(hparams)

                if hparams.is_procgen:

                    r = train_procgen(
                        agent=agent,
                        env_name=env_name,
                        seed=seed, 
                        n_envs=hparams.n_envs,
                        n_steps=hparams.n_steps,        
                        update_step=hparams.update_step,    
                        )

                else:
                    r = train(
                        agent=agent,
                        env_name=env_name,
                        seed=seed, 
                        n_envs=hparams.n_envs,
                        n_steps=hparams.n_steps,        
                        update_step=hparams.update_step,    
                        )

            rewards.append(r)

            plot_rewards(
                rewards,
                hparams.update_step,
                path=exp_path,
                env_name=env_name
                )

            save_results(
                env_name=env_name,
                rewards=rewards,
                path=exp_path,
                update_steps=hparams.update_step,
                seeds=hparams.random_seeds
                )

            save_model(agent, env_name, exp_path)


    save_arguments(path=exp_path, args=hparams)

def main():
    pass


if __name__ == "__main__":
    main()
