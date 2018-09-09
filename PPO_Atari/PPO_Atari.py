import os
import time
import argparse

import numpy as np
from threading import Lock


import tensorflow as tf

import gym
from gym import wrappers

from PPO import PPO
from Agent import Agent
from Atari_Runner import Atari_Runner

from Atari_util import INPUT_SHAPE, WINDOWS_SIZE

from Save_util import Save_file_util

# TensorFlowの警告を非表示に
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 環境名
ENV_NAME = "QbertNoFrameskip-v4"

# 入力サイズ
INPUT_SHAPE = (84, 84)

# 学習回数
NB_STAPS = 100000

# テスト回数
NB_TRY = 3

# PPO のパラメータ
TIMESTEPS = 256
CLIP_PARAM = 0.2
ENTCOEFF = 0.01
EPOCHS = 4
BATCHSIZE = 64
GAMMA = 0.99
LAMMD = 0.95

LEARNING_RATE = 0.0001

# ε-greedyのパラメータ
EPS_START = 0.5
EPS_END = 0.0
EPS_STEPS = 1000

def train(agent, runner):
    """学習"""

    # 学習実行
    runner.train(NB_STAPS)

    # モデル保存
    agent.save_brain(ENV_NAME)

    runner.test(NB_TRY)

def main():
    sess = tf.Session()
    env = gym.make(ENV_NAME)
    #print(env.action_space)
    ppo = PPO(sess=sess, input_shape=(*INPUT_SHAPE, WINDOWS_SIZE), nb_action=env.action_space.n, epochs=EPOCHS, batchsize=BATCHSIZE,\
                   timesteps=TIMESTEPS, clip_param=CLIP_PARAM, entcoeff=ENTCOEFF, gamma=GAMMA, lam=LAMMD, learning_rate=LEARNING_RATE)
    agent = Agent(ppo, env.action_space.n, EPS_START, EPS_END, EPS_STEPS)
    runner = Atari_Runner(env, agent)

    # tensorflow初期化
    sess.run(tf.global_variables_initializer())

    train(agent, runner)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=4)
    parser.add_argument('-t', '--timesteps', type=int, default=256)
    parser.add_argument('-b', '--batchsize', type=int, default=64)  
    parser.add_argument('-cp', '--clip_param', type=float, default=0.2)
    parser.add_argument('-ent', '--entcoeff', type=float, default=0.01)
    parser.add_argument('-g', '--gamma', type=float, default=0.99)
    parser.add_argument('-l', '--lammd', type=float, default=0.95)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001)
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('-n','--tryn', type=int, default=3)
    args = parser.parse_args()

    TIMESTEPS = args.timesteps
    CLIP_PARAM = args.clip_param
    ENTCOEFF = args.entcoeff
    EPOCHS = args.epoch
    BATCHSIZE = args.batchsize
    GAMMA = args.gamma
    LAMMD = args.lammd

    LEARNING_RATE = args.learning_rate

    NB_STAPS = args.steps
    NB_TRY = args.tryn

    Save_file_util.push(EPOCHS)
    Save_file_util.push(BATCHSIZE)
    Save_file_util.push(TIMESTEPS)
    Save_file_util.push(ENTCOEFF)
    Save_file_util.push(GAMMA)
    Save_file_util.push(LAMMD)
    Save_file_util.push(LEARNING_RATE)
    Save_file_util.push(NB_STAPS)

    main()