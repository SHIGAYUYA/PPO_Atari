import random
import numpy as np

class Agent:
    def __init__(self, brain, nb_action, eps_start, eps_end, eps_steps):
        self.brain = brain

        # Nステップ後までのadvantageを含む総報酬R
        self.R = 0.

        self.nb_action = nb_action
        
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_steps = eps_steps

    def act_with_v(self, s, frames):
        #"""学習時の行動"""
        ## ε-greedy法
        #if frames >= self.eps_steps:   
        #    eps = self.eps_end
        #else:
        #    eps = self.eps_steps + frames * (self.eps_end - self.eps_steps) / self.eps_steps  # linearly interpolate

        #s = np.reshape(s, (1, *s.shape))
        ##print(s.shape)

        #if random.random() < eps:
        #    _, v = self.brain.predict(s)
        #    # ランダムに行動
        #    return random.randint(0, self.nb_action - 1), v[0, 0]
        #else:
        #    p, v = self.brain.predict(s)
        #    #print(p.shape)
        #    a = np.random.choice(self.nb_action, p=p[0])
        #    return a, v[0, 0]

        p, v = self.brain.predict(s)
        u = np.random.uniform(size=p[0].shape)
        return np.argmax(p[0] - np.log(-np.log(u))), v[0, 0] 


    def act(self, s):
        """テスト時の行動"""
        #s = np.array([s])
        #p, v = self.brain.predict(s)
        #a = np.random.choice(NUM_ACTIONS, p=p[0])
        #return a
        p, v = self.brain.predict(s)
        u = np.random.uniform(size=p[0].shape)
        return np.argmax(p[0] - np.log(-np.log(u)))

    def update_parameter(self, next_v):
        self.brain.update_parameter(next_v)

    # brainのキューに追加するのみ
    def train_push(self, s, a, r, done, v):   
        self.brain.train_push(s, a, r, done, v)

    def get_learn_interval(self):
        return self.brain.get_learn_interval()

    def save_brain(self, target, path=None):
        """Brainの保存"""
        self.brain.save_model(target, path)

