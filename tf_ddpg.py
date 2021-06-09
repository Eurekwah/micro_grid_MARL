'''
Auther: Eurekwah
Date: 1970-01-01 08:00:00
LastEditors: Eurekwah
LastEditTime: 2021-06-10 03:45:59
FilePath: /code/tf_ddpg.py
'''
#coding=utf-8
import numpy as np
import tensorflow as tf
# import tensorflow.compat.v1 as tf
import math

import matplotlib
# import tensorlayer as tl


import time
import scipy
from scipy.integrate import odeint
import random
import os
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import env as env


# tf.compat.v1.disable_eager_execution()


np.random.seed(1)
tf.set_random_seed(1)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#####################  hyper parameters  ####################

STATE_DIM  = 4
ACTION_DIM = 3

ab=1 # action boundary

LR_A = 1e-5   # learning rate for actor
LR_C = 1e-5   # learning rate for critic
GAMMA = 0.99   # reward discount

TIMES = 3

REPLACEMENT = [
    dict(name='soft', tau=0.001),
    dict(name='hard', rep_iter_a=60, rep_iter_c=50)
][0]            # you can try different target replacement strategies origin set: 600,500
MEMORY_CAPACITY = 20000
BATCH_SIZE = 512

RENDER = False
OUTPUT_GRAPH = True

###############################  Actor  ####################################



class Actor(object):
    def __init__(self, sess, action_dim, action_bound, learning_rate, replacement):
        self.sess = sess
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.replacement = replacement
        self.t_replace_counter = 0

        with tf.variable_scope('Actor'):
            # input s, output a
            self.a = self._build_net(S, scope='eval_net', trainable=True)

            # input s_, output a, get a_ for critic
            self.a_ = self._build_net(S_, scope='target_net', trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')

        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            self.hard_replace = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        else:
            self.soft_replace = [tf.assign(t, (1 - self.replacement['tau']) * t + self.replacement['tau'] * e)
                                 for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 1)
            #init_b = tf.constant_initializer(0.1)
            init_b = tf.constant_initializer(0)
            net1 = tf.layers.dense(s, 64*TIMES, activation=tf.nn.sigmoid,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l1',
                                  trainable=trainable)

            net2 = tf.layers.dense(net1, 128*TIMES, activation=tf.nn.sigmoid,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                  trainable=trainable)

            net3 = tf.layers.dense(net2, 64*TIMES, activation=tf.nn.sigmoid,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l3',
                                  trainable=trainable)
            with tf.variable_scope('a'):
                actions = tf.layers.dense(net3, self.a_dim, activation=tf.nn.sigmoid, kernel_initializer=init_w,
                                          bias_initializer=init_b, name='a', trainable=trainable)
                scaled_a = tf.multiply(actions, self.action_bound, name='scaled_a')  # Scale output to -action_bound to action_bound
        return scaled_a

    def learn(self, s):   # batch update
        self.sess.run(self.train_op, feed_dict={S: s})

        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replace)
        else:
            if self.t_replace_counter % self.replacement['rep_iter_a'] == 0:
                self.sess.run(self.hard_replace)
            self.t_replace_counter += 1

    def choose_action(self, s):
        s = s[np.newaxis, :]    
        return self.sess.run(self.a, feed_dict={S: s})[0]  

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            # ys = policy;
            # xs = policy's parameters;
            # a_grads = the gradients of the policy to get more Q
            # tf.gradients will calculate dys/dxs with a initial gradients for ys, so this is dq/da * da/dparams
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)

        with tf.variable_scope('A_train'):
            opt = tf.train.AdamOptimizer(-self.lr)  # (- learning rate) for ascent policy
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))


###############################  Critic  ####################################

class Critic(object):
    def __init__(self, sess, STATE_DIM, ACTION_DIM, learning_rate, gamma, replacement, a, a_):
        self.sess = sess
        self.s_dim = STATE_DIM
        self.a_dim = ACTION_DIM
        self.lr = learning_rate
        self.gamma = gamma
        self.replacement = replacement

        with tf.variable_scope('Critic'):
            # Input (s, a), output q
            self.a = tf.stop_gradient(a)    # stop critic update flows to actor
            self.q = self._build_net(S, self.a, 'eval_net', trainable=True)

            # Input (s_, a_), output q_ for q_target
            self.q_ = self._build_net(S_, a_, 'target_net', trainable=False)    # target_q is based on a_ from Actor's target_net

            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        with tf.variable_scope('target_q'):
            self.target_q = R + self.gamma * self.q_

        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))
            
        with tf.variable_scope('C_train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, self.a)[0]   # tensor of gradients of each sample (None, a_dim)

        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            self.hard_replacement = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        else:
            self.soft_replacement = [tf.assign(t, (1 - self.replacement['tau']) * t + self.replacement['tau'] * e)
                                     for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.1)
            init_b = tf.constant_initializer(0.1)

            with tf.variable_scope('l1'):
                n_l1 = 30
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)

            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)   # Q(s,a)
        return q

    def learn(self, s, a, r, s_):
        self.sess.run(self.train_op, feed_dict={S: s, self.a: a, R: r, S_: s_})
        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replacement)
        else:
            if self.t_replace_counter % self.replacement['rep_iter_c'] == 0:
                self.sess.run(self.hard_replacement)
            self.t_replace_counter += 1

#####################  Memory  ####################

class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]
'''
class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0
        self.dt = np.dtype([('index', int), ('reward', float)])
        self.stack = np.zeros((capacity, 1), dtype=self.dt)

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        if self.pointer < self.capacity:
            index = self.pointer  # replace the old memory with new memory
            self.stack[index] = np.array([(index, r)], dtype=self.dt)
            self.data[index, :] = transition
        else:
            index = np.argmin(self.stack[:]['reward'])
            if r >= self.stack[index]['reward']:
                self.stack[index] = np.array([(index, r)], dtype=self.dt)
                self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]
'''
####################  main  ######################
with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='s')
with tf.name_scope('R'):
    R = tf.placeholder(tf.float32, [None, 1], name='r')
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='s_')

def train(action_bound, episodes):
    sess = tf.Session()
    bound = action_bound
    actor = Actor(sess, ACTION_DIM, bound, LR_A, REPLACEMENT)
    critic = Critic(sess, STATE_DIM, ACTION_DIM, LR_C, GAMMA, REPLACEMENT, actor.a, actor.a_)
    actor.add_grad_to_graph(critic.a_grads)
    sess.run(tf.global_variables_initializer())

    M = Memory(MEMORY_CAPACITY, dims=2 * STATE_DIM + ACTION_DIM + 1)

    environment = env.Env()
    reward_temp = []    # 过程reward
    a0_stack = []       # 储能装置功率
    a1_stack = []       # 充电补贴
    a2_stack = []       # 柴油机功率
    ev_stack = []       # 充电补贴后负荷
    pw_stack = []       # 输出电力
    rw_stack = []       # 每小时reward
    pc_stack = []       # 电价
    wt_stack = []       # wait number
    st_stack = []       # storage unit
    var = 4
    
    for i in range(episodes):
        s = environment.reset()
        episode_reward = 0  # 每代reward

        for time in range(24):
            a = actor.choose_action(s)
            a = np.clip(np.random.normal(a, var), -1, 1)
            s1, r1, crt_price = environment.step(time, a)
            M.store_transition(s, a, r1, s1)
            s = s1
            
            if M.pointer > MEMORY_CAPACITY:
                var *= 0.999
                b_M = M.sample(BATCH_SIZE)
                b_s = b_M[:, :STATE_DIM]
                b_a = b_M[:, STATE_DIM: STATE_DIM + ACTION_DIM]
                b_r = b_M[:, -STATE_DIM - 1: -STATE_DIM]
                b_s_ = b_M[:, -STATE_DIM:]
                critic.learn(b_s, b_a, b_r, b_s_)
                actor.learn(b_s)
                if i == episodes - 1:
                    a0_stack.append(a[0])
                    a1_stack.append(a[1])
                    a2_stack.append(a[2])
                    ev_stack.append(s[0])
                    pw_stack.append(s[1])
                    pc_stack.append(crt_price)
                    rw_stack.append(r1)
                    st_stack.append(s[2])
                    wt_stack.append(s[3])
            
        # print(episodes, ":", episode_reward)
            episode_reward += r1
        reward_temp.append(episode_reward)
            
        if i == episodes - 1:
            fig = plt.figure(figsize=(16, 16), dpi=300)
            a1 = plt.subplot2grid((4, 3), (0, 0))
            a1.set_ylim([-1.5, 1.5])
            a1.plot(a0_stack)
            a1.set_title('storage unit power')
            a2 = plt.subplot2grid((4, 3), (0, 1))
            a2.set_ylim([-1.5, 1.5])
            a2.plot(a1_stack)
            a2.set_title('charging subsidies')
            a3 = plt.subplot2grid((4, 3), (0, 2))
            a3.set_ylim([-1.5, 1.5])
            a3.plot(a2_stack)
            a3.set_title('diesel engine power')
            a4 = plt.subplot2grid((4, 3), (1, 0))
            a4.plot(ev_stack, label='real')
            a4.plot([105, 310, 865, 2150, 3580, 4670, 4765, 3780, 2410, 1340, 610, 230, 90, 35, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0], label='origin')
            a4.plot(pw_stack, label='output power')
            a4.set_title('load')   
            a6 = plt.subplot2grid((4, 3), (1, 1))
            a6.plot(rw_stack)
            a6.set_title('reward per hour')
            a7 = plt.subplot2grid((4, 3), (1, 2))
            a7.set_ylim([-0.5, 2.5])
            a7.plot(pc_stack)
            a7.set_title('electricity price')
            a8 = plt.subplot2grid((4, 3), (3, 0), colspan=3)
            a8.plot(reward_temp)
            a8.set_title('reward')
            a9 = plt.subplot2grid((4, 3), (2, 0), colspan=2)
            a9.plot(wt_stack)
            a9.set_title('wait number')
            aA = plt.subplot2grid((4, 3), (2, 2))
            aA.plot(st_stack)
            aA.set_title('storage unit')
            aA.set_ylim([0, 4000])
            fig.tight_layout()
            fig.savefig('/public/home/zyc20000201/code/ddpg/1.png')

if __name__ == '__main__':
    starttime = time.time()
    train([1, 1, 1], 3000)
    endtime = time.time()
    print("time:", round(endtime - starttime, 2), "secs")