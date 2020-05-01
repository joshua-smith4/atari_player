from dnn.agent import QLearningAgent
import gym

import os,sys
import argparse
import copy

import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--keep_training', type=int, default=1)
parser.add_argument('--epochs', type=int, default=1000)
args = parser.parse_args()
path_to_saved_model = 'saved_model/latest';

if __name__ == '__main__':
    env = gym.make('Breakout-v0') 
    qagent = QLearningAgent(
            env.observation_space.shape, env.action_space.n)
    if args.keep_training == 1:
        print('loading network from previous training')
        qagent.network = tf.keras.models.load_model(path_to_saved_model)
    for i in range(args.epochs):
        print('Epoch {}'.format(i))
        obs = env.reset()
        obs = obs / 255.0
        done = False
        max_reward = 0;
        while not done:
            act = qagent.act(obs, epsilon=qagent.epsilon)
            obs_next, reward, done, _ = env.step(act)
            obs_next = obs_next / 255.0
            if reward > max_reward: max_reward = reward
            act = tf.keras.utils.to_categorical(act, env.action_space.n)
            qagent.q_learning_step(
                    obs, act, reward, obs_next, done, qagent.batch_size)
            obs = obs_next
        print('max_reward: {}'.format(max_reward))
        qagent.network.save('saved_model/latest')

