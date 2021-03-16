'''
Auther: Eurekwah
Date: 1970-01-01 08:00:00
LastEditors: Eurekwah
LastEditTime: 2021-03-09 03:50:31
FilePath: /code/tf_ddpg.py
'''
import numpy as np
import tensorflow.compat.v1 as tf
import math
#import tensorlayer as tl
import time
import scipy
from scipy.integrate import odeint
import random
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import new_env as env

tf.compat.v1.disable_eager_execution()

np.random.seed(1)
tf.set_random_seed(1)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#####################  hyper parameters  ####################

STATE_DIM  = 4
ACTION_DIM = 3

ab=1 # action boundary

LR_A = 1e-5    # learning rate for actor
LR_C = 1e-5   # learning rate for critic
GAMMA = 0.3     # reward discount


REPLACEMENT = [
    dict(name='soft', tau=0.001),
    dict(name='hard', rep_iter_a=600, rep_iter_c=500)
][0]            # you can try different target replacement strategies origin set: 600,500
MEMORY_CAPACITY = 15000
BATCH_SIZE = 256

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
            net1 = tf.layers.dense(s, 64, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l1',
                                  trainable=trainable)

            net2 = tf.layers.dense(net1, 128, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                  trainable=trainable)

            net3 = tf.layers.dense(net2, 64, activation=tf.nn.relu,
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
    reward_temp = []
    #plt.ion()
    var = 2
    
    for i in range(episodes):
        s = environment.reset()
        episode_reward = 0

        a0_stack = []
        a1_stack = []
        a2_stack = []
        ev_stack = []
        price_stack = []
        reward_stack = []

        state_stack = []
        for time in range(24):
            a = actor.choose_action(s)
            a = np.clip(np.random.normal(a, var), -1, 1)
            s1, r1 = environment.step(time, a)
            M.store_transition(s, a, r1, s1)
            s = s1
            reward_stack.append(r1)
            if M.pointer > MEMORY_CAPACITY:
                var *=0.9995
                b_M = M.sample(BATCH_SIZE)
                b_s = b_M[:, :STATE_DIM]
                b_a = b_M[:, STATE_DIM: STATE_DIM + ACTION_DIM]
                b_r = b_M[:, -STATE_DIM - 1: -STATE_DIM]
                b_s_ = b_M[:, -STATE_DIM:]
                a0_stack.append(a[0])
                a1_stack.append(a[1])
                a2_stack.append(a[2])
                ev_stack.append(s[0])
                state_stack.append(s[3])
                critic.learn(b_s, b_a, b_r, b_s_)
                actor.learn(b_s)
            episode_reward += r1
        #if M.pointer > MEMORY_CAPACITY:
        reward_temp.append(episode_reward)
        print(i, ': ', episode_reward, 'exploration:', var)
        plt.clf()
        plt.subplot(321)
        plt.plot(reward_temp)
        plt.subplot(322)
        # plt.ylim(-1, 1)
        plt.plot(a0_stack, label='Price subsidy')
        plt.plot(a1_stack, label='Energy storage unit power')
        plt.plot(a2_stack, label='Diesel power')
        plt.legend()
        plt.subplot(323)
        plt.plot([55, 165, 430, 1060, 1700, 2075, 1980, 1370, 720, 320, 105, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10], label='origin')
        plt.plot(ev_stack, label='with price subsidy')
        plt.legend()
        plt.subplot(324)
        plt.plot(state_stack)
        plt.subplot(325)
        plt.plot(reward_stack)
            #plt.subplot(326)
            
            #plt.pause(0.05)
    #plt.ioff()
    #plt.show()
    plt.savefig('/public/home/zyc20000201/code/ddpg/figure13.png')

if __name__ == '__main__':
    train([1, 1, 1], 2000)