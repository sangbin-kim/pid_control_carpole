"""
Enviroment of problem to be solved.
"""
import gym


class CartPoleEnv(object):
    @staticmethod
    def make():
        return gym.make('CartPole-v0')