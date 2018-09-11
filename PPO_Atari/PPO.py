import numpy as np


from keras import Sequential
from keras.layers import Flatten, Dense, Conv2D, Input
from keras.optimizers import Adam
from keras.utils import np_utils, plot_model
from keras import backend as K
from keras.models import load_model, Model

import tensorflow as tf
import keras.backend as K

from sklearn.utils import shuffle

from Save_util import Save_file_util

class PPO:
    def __init__(self, sess, nb_action, input_shape, epochs, batchsize, timesteps, clip_param, entcoeff, gamma, lam, learning_rate, test=false):
        self.sess = sess

        # 学習の定数
        self.epochs = epochs
        self.batchsize = batchsize
        self.timestaps = timesteps
        self.clip_param = clip_param
        self.entcoeff = entcoeff
        self.gamma = gamma
        self.lam = lam

        K.set_session(sess)

        if test:
            self.model = load_model(Save_file_util.get_file_name("ppo_weight_"+ target, '.h5f'))
        else:
            with tf.name_scope("BrainOld"):
                self.model_old = self._build_model(input_shape, nb_action)

            with tf.name_scope("brain"):
                # ニューラルネットワークの形を決定
                self.model = self._build_model(input_shape, nb_action)  

                # loss関数を最小化していくoptimizerの定義
                self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate)  

                # ネットワークの学習やメソッドを定義
                self.graph = self.build_graph(input_shape, nb_action)  
        
            # 重みのコピー
            self.model_old.set_weights([w for w in self.model.get_weights()])

            # モデルの可視化
            plot_model(self.model, to_file='PPO.png', show_shapes=True)

            # キューの初期化
            self.train_queue = {"ob" : [], "rew" : [], "vpred" : [], "nonterminal" : [],"ac" : []}


    def _build_model(self, input_shape, nb_action):     # Kerasでネットワークの形を定義
        #print(input_shape)
        l_input = Input(shape=input_shape)
        l_cnn = Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4), activation="relu")(l_input)
        l_cnn = Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), activation="relu")(l_cnn)
        l_cnn = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation="relu")(l_cnn)
        l_flat = Flatten()(l_cnn)
        l_dense = Dense(512, activation='relu')(l_flat)
        out_actions = Dense(nb_action, activation='linear')(l_dense)
        out_value = Dense(1, activation='linear')(l_dense)
        model = Model(inputs=[l_input], outputs=[out_actions, out_value])

        # Keras の実行準備
        model._make_predict_function()  

        return model

    def build_graph(self, input_shape, nb_action):      # TensorFlowでネットワークの重みをどう学習させるのかを定義します
        self.s_t = tf.placeholder(tf.float32, shape=(None, *input_shape))
        self.a_t = tf.placeholder(tf.float32, shape=(None, nb_action))
        self.atarg = tf.placeholder(dtype=tf.float32, shape=[None])
        self.vtarg = tf.placeholder(dtype=tf.float32, shape=[None])

        p, v = self.model(self.s_t)

        p_old, _ = self.model_old(self.s_t)

        # loss関数を定義
        # p側
        #ratio = tf.div(p * self.a_t, p_old * self.a_t + K.epsilon())
        #advantage_CPI = tf.reduce_sum(ratio) * self.atarg
        #clipped_advantage_CPI = tf.clip_by_value(tf.reduce_sum(ratio), 1.0 - self.clip_param, 1.0 + self.clip_param) * self.atarg 
        #loss_CLIP = -tf.reduce_mean(tf.minimum(advantage_CPI, clipped_advantage_CPI))

        ## v側
        #loss_value = tf.reduce_mean(tf.square(v - self.vtarg))

        ## entropy 最大化の正則化
        #entropy = - tf.reduce_mean(p * tf.log(p + K.epsilon()))  
        #loss_ent = (- self.entcoeff) * entropy

        ## 最終的なloss関数
        #self.loss_total = loss_CLIP + loss_value + loss_ent

        ## 求めた勾配で重み変数を更新する定義
        #minimize = self.opt.minimize(self.loss_total)   
        def entropy(logits):
            a0 = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
            ea0 = tf.exp(a0)
            z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
            p0 = ea0 / z0
            return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)

        ent = entropy(p)
        meanent = tf.reduce_mean(ent)
        pol_entpen = (-self.entcoeff) * meanent

        def logp(a, logits):
            return -tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=logits,
            labels=a)

        ratio = tf.exp(logp(self.a_t, p) - logp(self.a_t, tf.stop_gradient(p_old))) # pnew / pold
        surr1 = ratio * self.atarg # surrogate from conservative policy iteration
        surr2 = tf.clip_by_value(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * self.atarg #
        pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
        vf_loss = tf.reduce_mean(tf.square(v - self.vtarg))
        total_loss = pol_surr + pol_entpen + vf_loss

        minimize = self.opt.minimize(total_loss) 
        return minimize


    def update_parameter(self, next_v):     
        """重みを学習・更新"""
        if len(self.train_queue["ob"]) < self.timestaps:    # データがたまっていない場合は更新しない
            #print(len(self.train_queue["ob"]) )
            return

        # アドバンテージの計算
        ob, ac, atarg, vtarg = self.get_atarg_and_vtarg(next_v)
        self.train_queue = {"ob" : [], "rew" : [], "vpred" : [], "nonterminal" : [],"ac" : []}  

        atarg = (atarg - atarg.mean()) / atarg.std() 

        N_train = len(atarg)
        n_batches = N_train // self.batchsize

        for epoch in range(self.epochs):
            ob_, ac_, atarg_, vtarg_ = shuffle(ob, ac, atarg, vtarg)

            for i in range(self.batchsize):
                start = i * self.batchsize
                end = start + self.batchsize

                # 重みの更新に使用するデータ
                feed_dict = {self.s_t: ob[start:end, :, :, :], self.a_t: ac_[start:end, :], self.atarg: atarg_[start:end], self.vtarg: vtarg_[start:end]}

                # Brainの重みの更新
                minimize = self.graph
                self.sess.run(minimize, feed_dict)

        # 重みのコピー
        self.model_old.set_weights([w for w in self.model.get_weights()])


    def predict(self, s):    # 状態sから各actionの確率pと状態価値vを返す
        p, v = self.model.predict(np.array([s]))
        return p, v

    def train_push(self, s, a, r, done, v):
        self.train_queue["ob"].append(s)

        self.train_queue["ac"].append(a)

        self.train_queue["rew"].append(r)

        if done:
            self.train_queue["vpred"].append(0)
            self.train_queue["nonterminal"].append(0.)
        else:
            self.train_queue["vpred"].append(v)
            self.train_queue["nonterminal"].append(1.)

    def get_atarg_and_vtarg(self, next_v):
        # 最後のvは終端ではないものとして計算
        nonterminal = np.append(self.train_queue["nonterminal"], 1) 
        vpred = np.append(self.train_queue["vpred"], next_v)

        # timestep分スタックされている
        T = len(self.train_queue["rew"])

        atarg = gaelam = np.empty(T, 'float32')
        rew = self.train_queue["rew"]
        lastgaelam = 0
        for t in reversed(range(T)):
            delta = rew[t] + self.gamma * vpred[t+1] * nonterminal[t+1] - vpred[t]
            atarg[t] = lastgaelam = delta + self.gamma * self.lam * nonterminal[t+1] * lastgaelam

        vtarg = atarg + np.array(self.train_queue["vpred"])

        return np.array(self.train_queue["ob"]), np.array(self.train_queue["ac"]), atarg, vtarg

    def save_model(self, path=None):
        """モデルの重み保存"""
        if path is None:
            p = Save_file_util.get_file_name("ppo_weight", '.h5f')
        else:
            p = path
        #print(p)
        self.model.save(p)

    def get_learn_interval(self):
        return self.timestaps