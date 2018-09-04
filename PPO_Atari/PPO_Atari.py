import os
import time
import argparse

import numpy as np

from keras import Sequential
from keras.callbacks import TensorBoard
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils, plot_model
from keras import backend as K
from keras.models import load_model

import gym
from gym import wrappers

# TensorFlowの警告を非表示に
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 環境名
ENV_NAME = "QbertNoFrameskip-v4"

# 入力サイズ
INPUT_SHAPE = (84, 84)

# 学習が終了したことを示すフラグ
isLearned = False

# スレッドの数
N_WORKERS = 8

# 各スレッドの更新ステップ間隔
LEARN_INTERVAL = 30

# 学習回数
NB_STAPS = 1750000

def train():
    # TensorFlowのセッション開始
    sess = tf.Session()

    # TensorFlowでマルチスレッドにするための準備
    coord = tf.train.Coordinator()

    # スレッドの作成
    with tf.device("/cpu:0"):
        ppo = PPO()     # ディープニューラルネットワークのクラスです
        threads = []     # 並列して走るスレッド
        # 学習するスレッドを用意
        for i in range(N_WORKERS):
            thread_name = "local_thread"+str(i+1)
            threads.append(Worker_thread(thread_name=thread_name, thread_type="learning", brain=brain))

        # 学習後にテストで走るスレッドを用意
        threads.append(Worker_thread(thread_name="test_thread", thread_type="test", brain=brain))

    # TensorFlowでマルチスレッドを実行
    sess.run(tf.global_variables_initializer())

    running_threads = []
    for worker in threads:
        job = lambda: worker.run()
        t = threading.Thread(target=job)
        t.start()

    # スレッドの終了の同期
    coord.join(running_threads)

    # モデル保存
