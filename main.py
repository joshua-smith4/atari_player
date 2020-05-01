from dnn.agent import QLearningAgent
import gym

import os,sys
import argparse

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
    nop = np.array([1,0,0,0,0])
    for i in range(args.epochs):
        obs_n = env.reset()
        done_master = [False for i in range(len(obs_n))]
        print('Epoch {}'.format(i))
        count = 0
        while not all(done_master) and count < args.max_frames:
            count += 1
            act_n = []
            for k in range(len(obs_n)):
                if done_master[k]:
                    act_n.append(nop)
                    continue
                act_n.append(qagent.act(obs_n[k], qagent.epsilon)) 
            obs_next, reward_n, done_n, _ = env.step(act_n)
            for k in range(len(obs_n)):
                if done_master[k]: continue
                if done_n[k]: done_master[k] = True
                qagent.q_learning_step(obs_n[k], act_n[k], reward_n[k], obs_next[k], done_n[k], qagent.batch_size)
        if all(done_master):
            print('all the goals were reached')
        qagent.network.save('saved_model/latest')

