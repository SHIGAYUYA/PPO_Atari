from threading import Lock
from collections import deque

from PPO_Atari import LEARN_INTERVAL, NB_STAPS
from Atari_util import preprocess, WINDOWS_SIZE

class Environment:
    frames = 0 # セッション全体での試行数

    def __init__(self, name, thread_type, brain, coord):
        self.name = name
        self.thread_type = thread_type

        # テストは動画保存
        if self.thread_type is 'test':
            env = wrappers.Monitor(env, './movie_folder', video_callable=(lambda ep: True), force=True)
        else:
            self.env = gym.make(ENV)

        # 環境内で行動するagent
        self.agent = Agent(brain)

        # スレッド制御クラス
        self.coord = coord 

        # framesの排他制御
        self.lock = Lock()

        # このスレッドの総試行回数
        self.count_trial_each_thread = 0

        # 状態用メモリ
        recent_memory = deque([], maxlen=WINDOWS_SIZE)

    def _run(self, a):
        R = 0
        step = 0

        # 行動をWINDOW_SIZE分蓄積
        while len(recent_memory) != WINDOWS_SIZE:
            # 行動を実施
            _s, r, done, info = self.env.step(a)
            step += 1

            # 報酬の和
            R += r

            # 全体の行動回数を増やす
            with self.lock:
                Environment.frames += 1

            if done:  # terminal state
                break

            recent_memory.append(_s)

         # 状態を前処理
        s = preprocess([recent_memory.popleft() for _ in range(WINDOWS_SIZE)])

        return s, R, done, info, step

    def run(self):
        # 環境初期化
        recent_memory.append(self.env.reset())
        a = env.action_space.sample()

        # 初期状態を前処理
        s, r, _, _ = self._run(a)
        
        # 累積報酬初期化
        R = r

        # 実効ステップ
        step = 0

        while True:
            # テストでは描画
            if self.thread_type is 'test':
                self.env.render()   
                time.sleep(0.1)

            # 行動を決定
            a = self.agent.act(s)

            s_, r, done, info, _step = self._run(a)
            step += _step

            if done:  # terminal
                s_ = None

            # 報酬と経験を、Brainにプッシュ
            self.agent.advantage_push_brain(s, a, r, s_)

            # 状態更新
            s = s_

            # 報酬累積
            R += r

            # 終了時がTmaxごとに、parameterServerの重みを更新
            if done or (frames % LEARN_INTERVAL == 0):  
                if self.thread_type is 'learning':
                    self.agent.brain.update_parameter_server()

            if done:
                self.count_trial_each_thread += 1  # このスレッドの総試行回数を増やす
                break

        # 総試行数、スレッド名、今回の報酬を出力
        print("スレッド："+self.name + " 試行数："+str(self.count_trial_each_thread) + " 今回のステップ:" + str(step) + " 今回の累積報酬:" + str(R))

        # 全体で行動回数が規定回数に達したら終了
        if frames > NB_STAPS:
            # 実行中の全スレッド停止
            self.coord.request_stop()
