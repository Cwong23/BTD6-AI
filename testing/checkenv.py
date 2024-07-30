from stable_baselines3.common.env_checker import check_env
from neuralNetwork import BTDEnv

env = BTDEnv()

check_env(env)