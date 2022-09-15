#!/usr/bin/python

import gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
import simple_driving
import time

import sys

def main() -> None :

    # Create environment
    env = gym.make('SimpleDriving-v0')

    # Instantiate the agent
    model = PPO('MlpPolicy', env, verbose=1)
    # Train the agent
    model.learn(total_timesteps=int(2e5))
    # Save the agent
    model.save("dqn_SimpleDriving")
    del model  # delete trained model to demonstrate loading

    # Load the trained agent
    # NOTE: if you have loading issue, you can pass `print_system_info=True`
    # to compare the system on which the model was trained vs the current one
    # model = DQN.load("dqn_lunar", env=env, print_system_info=True)
    model = DQN.load("dqn_SimpleDriving", env=env)

    # Evaluate the agent
    # NOTE: If you use wrappers with your environment that modify rewards,
    #       this will be reflected here. To evaluate with original rewards,
    #       wrap environment in a "Monitor" wrapper before other wrappers.
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

    # Enjoy trained agent
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        env.render()

if __name__ == '__main__':
    main()
