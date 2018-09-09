from collections import deque

import numpy as np

import gym
from gym import wrappers

from Atari_util import preprocess, WINDOWS_SIZE

from Save_util import Save_file_util

class Atari_Runner:
    def __init__(self, env, agent):
        # 環境格納
        self.env = env

        self.nb_action = env.action_space.n

        # 環境内で行動するagent
        self.agent = agent

        # 状態用メモリ
        self.recent_memory = deque([], maxlen=WINDOWS_SIZE)

        self.frames = 0
        self.count_trial = 0

    def _run(self, a):
        R = 0
        step = 0

        # 行動をWINDOW_SIZE分フレームスキップ
        for _ in range(WINDOWS_SIZE):
            # 行動を実施
            _s, r, done, info = self.env.step(a)
            step += 1

            # 報酬の和
            R += r

            self.recent_memory.append(_s)

            if done:  # terminal state
                break


        # 状態を前処理
        s = preprocess(list(self.recent_memory), WINDOWS_SIZE, tol=False)

        return s, R, done, info, step

    def run(self, nb_steps=None):
        # 環境初期化
        self.recent_memory.append(self.env.reset())
        a = self.env.action_space.sample()

        # 初期状態を前処理
        s, r, _, _, _ = self._run(a)
        
        # 累積報酬初期化
        R = r

        # 実効ステップ
        step = 0

        while True:
            # テストでは描画
            if self.mode is 'test':
                self.env.render()   

            # 行動を決定
            if self.mode is 'train':
                a, v = self.agent.act_with_v(s, self.frames)
            elif self.mode is 'test':
                a = self.agent.act(s)

            s_, r, done, info, _step = self._run(a)
            step += _step

            if self.mode is 'train':
                # learn_intervalごとに重みを更新
                if self.frames > 0 and self.frames % self.learn_interval == 0:
                    self.agent.update_parameter(v)

                # one-hot
                ac = np.zeros(self.nb_action)
                ac[a] = 1

                # 報酬と経験をプッシュ
                self.agent.train_push(s, ac, r, done, v)

            # 全体の行動回数を増やす
            self.frames += 1

            # 状態更新
            s = s_

            # 報酬累積
            R += r

            

            if done:
                self.count_trial += 1
                break

        # 総試行数、スレッド名、今回の報酬を出力
        print("試行数：" + str(self.count_trial) + " 今回のステップ:" + str(step) + " 今回の累積報酬:" + str(R))
        if self.mode is 'train':
            print("進行度：" + str(self.frames) + '/' + str(nb_steps))
        #elif self.mode is 'test':
        #    Save_file_util.create_result([self.count_trial, step, R])
        

    def train(self, nb_steps):

        # mode指定
        self.mode = 'train'

        # 重み更新頻度
        self.learn_interval = self.agent.get_learn_interval()

        # 初期化
        self.frames = 0
        self.count_trial = 0

        # 全体で行動回数が規定回数に達したら終了
        while self.frames < nb_steps:
            self.run(nb_steps)

    def test(self, nb_try=1):
        # 環境宣言 動画保存
        self.env = wrappers.Monitor(self.env, Save_file_util.get_file_name('./movie_folder'), video_callable=(lambda ep: True), force=True)

        # mode指定
        self.mode = 'test'

        # 初期化
        self.frames = 0
        self.count_trial = 0

        # 指定回数繰り返し
        for _ in range(nb_try):
            self.run()