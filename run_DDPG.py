from environment import Environment

import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import pandas as pd
import  tensorflow as tf


with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape=[None, 300], name='s')
with tf.name_scope('R'):
    R = tf.placeholder(tf.float32, [None, 1], name='r')
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, 300], name='s_')

sess = tf.Session()

class Actor(object):
    def __init__(self, sess, action_dim, learning_rate, replacement,e_greedy_increment=0.01, e_greedy = 0.95):
        self.sess = sess
        self.a_dim = action_dim

        self.lr = learning_rate
        self.replacement = replacement
        self.t_replace_counter = 0
        self.epsilon_max = e_greedy

        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max



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
            init_w = tf.random_normal_initializer(0., 0.1)
            init_b = tf.constant_initializer(0.8)
            #net = tf.layers.dense(s, 100, activation=tf.nn.relu,
            #                      kernel_initializer=init_w, bias_initializer=init_b, name='l1',
            #                      trainable=trainable)

            #with tf.variable_scope('q1'):
            #    q1 = tf.layers.dense(net, 256, activation=tf.nn.relu,kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)   # Q(s,a)
            #with tf.variable_scope('q2'):
            #    q2 = tf.layers.dense(q1, 128, activation=tf.nn.relu,kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)   # Q(s,a)

            #with tf.variable_scope('q'):
            #   q3 = tf.layers.dense(q2, 64, kernel_initializer=init_w, activation=tf.nn.relu, bias_initializer=init_b, trainable=trainable)   # Q(s,a)\

            #with tf.variable_scope('a'):
            #    actions = tf.layers.dense(net, self.a_dim, activation=tf.nn.relu, kernel_initializer=init_w,
            #                              bias_initializer=init_b, name='a', trainable=trainable)

            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [300, 512], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, 512], initializer=init_b, trainable=trainable)
                l1 = tf.nn.softmax(tf.matmul(s, w1) + b1)
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [512, 256], initializer=init_w, trainable=trainable)
                b2 = tf.get_variable('b2', [1, 256], initializer=init_b, trainable=trainable)
                l2 = tf.matmul(l1, w2) + b2

            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [256, 128], initializer=init_w, trainable=trainable)
                b3 = tf.get_variable('b3', [1, 128], initializer=init_b, trainable=trainable)
                l3 = tf.matmul(l2, w3) + b3

            with tf.variable_scope('l4'):
                w4 = tf.get_variable('w4', [128, 64], initializer=init_w, trainable=trainable)
                b4 = tf.get_variable('b4', [1, 64], initializer=init_b, trainable=trainable)
                l4 = tf.matmul(l3, w4) + b4

            with tf.variable_scope('l5'):
                w5 = tf.get_variable('w5', [64, self.a_dim], initializer=init_w, trainable=trainable)
                b5 = tf.get_variable('b5', [1, self.a_dim], initializer=init_b, trainable=trainable)
                actions = tf.matmul(l4, w5) + b5
            with tf.variable_scope('a'):
                action = tf.layers.dense(actions, self.a_dim, activation=tf.nn.softmax, kernel_initializer=init_w,
                                         bias_initializer=init_b, name='a', trainable=trainable)

        return action

    def learn(self, s):  # batch update
        self.sess.run(self.train_op, feed_dict={S: s})

        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replace)
        else:
            if self.t_replace_counter % self.replacement['rep_iter_a'] == 0:
                self.sess.run(self.hard_replace)
            self.t_replace_counter += 1
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        print("epslion:",self.epsilon)

    def choose_action(self, s,var):
        s = s[np.newaxis, :]  # single state
        action = self.sess.run(self.a, feed_dict={S: s})  # single action
        #print(action)

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            #action = np.clip(np.random.normal(action, var), 0, 14)
            action = np.argmax(action)
        else:
            action = np.random.randint(0, self.a_dim)
        return action



    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            # ys = policy;
            # xs = policy's parameters;
            # a_grads = the gradients of the policy to get more Q
            # tf.gradients will calculate dys/dxs with a initial gradients for ys, so this is dq/da * da/dparams
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)

        with tf.variable_scope('A_train'):
            opt = tf.train.GradientDescentOptimizer(-self.lr)  # (- learning rate) for ascent policy

            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))

    ###############################  Critic  ####################################


class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, replacement, a, a_):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.replacement = replacement



        with tf.variable_scope('Critic'):
            # Input (s, a), output q
            self.a = tf.stop_gradient(a)  # stop critic update flows to actor
            self.q = self._build_net(S, self.a, 'eval_net', trainable=True)

            # Input (s_, a_), output q_ for q_target
            self.q_ = self._build_net(S_, a_, 'target_net',
                                      trainable=False)  # target_q is based on a_ from Actor's target_net

            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        with tf.variable_scope('target_q'):
            self.target_q = R + self.gamma * self.q_

        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

        with tf.variable_scope('C_train'):
            self.train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)

        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, self.a)[0]  # tensor of gradients of each sample (None, a_dim)

        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            self.hard_replacement = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        else:
            self.soft_replacement = [
                tf.assign(t, (1 - self.replacement['tau']) * t + self.replacement['tau'] * e)
                for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s, a, scope, trainable):

        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.1)
            init_b = tf.constant_initializer(0.8)

            with tf.variable_scope('l1'):
                n_l1 = 512
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)

            #with tf.variable_scope('q1'):
            #    q1 = tf.layers.dense(net, 256, activation=tf.nn.relu, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)   # Q(s,a)
            #with tf.variable_scope('q2'):
            #    q2 = tf.layers.dense(q1, 128, activation=tf.nn.relu,kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)   # Q(s,a)

            #with tf.variable_scope('q3'):
            #    q3 = tf.layers.dense(q2, 64, kernel_initializer=init_w, activation=tf.nn.relu, bias_initializer=init_b, trainable=trainable)   # Q(s,a)

            #with tf.variable_scope('q5'):
            #    q5 = tf.layers.dense(net, 1, kernel_initializer=init_w, activation=tf.nn.relu, bias_initializer=init_b, trainable=trainable)   # Q(s,a)


            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, 256], initializer=init_w, trainable=trainable)
                b2 = tf.get_variable('b2', [1, 256], initializer=init_b, trainable=trainable)
                l2 = tf.matmul(net, w2) + b2

            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [256, 128], initializer=init_w, trainable=trainable)
                b3 = tf.get_variable('b3', [1, 128], initializer=init_b, trainable=trainable)
                l3 = tf.matmul(l2, w3) + b3

            with tf.variable_scope('l4'):
                w4 = tf.get_variable('w4', [128, 64], initializer=init_w, trainable=trainable)
                b4 = tf.get_variable('b4', [1, 64], initializer=init_b, trainable=trainable)
                l4 = tf.matmul(l3, w4) + b4

            with tf.variable_scope('l5'):
                w5 = tf.get_variable('w5', [64, 1], initializer=init_w, trainable=trainable)
                b5 = tf.get_variable('b5', [1, 1], initializer=init_b, trainable=trainable)
                q5 = tf.matmul(l4, w5) + b5
            #with tf.variable_scope('q'):
            #    q = tf.layers.dense(q_eval, 1, kernel_initializer=init_w, bias_initializer=init_b,
            #                             trainable=trainable)  # Q(s,a)

        return q5

    def learn(self, s, a, r, s_):

        #self.sess.run(self.train_op, feed_dict={S: s, self.a: a, R: r, S_: s_})
        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replacement)
        else:
            if self.t_replace_counter % self.replacement['rep_iter_c'] == 0:
                self.sess.run(self.hard_replacement)
            self.t_replace_counter += 1
        _, self.cost = self.sess.run([self.train_op, self.loss],
                                     feed_dict={S: s, self.a: a, R: r, S_: s_})
        dict_bp = {'loss': self.cost}
        df = pd.DataFrame(dict_bp, index=[0])
        df.to_csv("loss_DDPG.csv", sep=',', header=None, mode='a', index=False)


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
        #assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]





def run_DDPG(item,Candidate_Paths):
    env = Environment(10,15,42)
    s_dim = env.n_features
    a_dim = env.n_actions
    #a_bound = env.action_space.high
    MEMORY_CAPACITY = 1000
    BATCH_SIZE = 32
    LR_A = 0.000001  # learning rate for actor
    LR_C = 0.000001  # learning rate for critic
    GAMMA = 0.9  # reward discount
    REPLACEMENT = [
        dict(name='soft', tau=0.01),
        dict(name='hard', rep_iter_a=600, rep_iter_c=500)
    ][0]  # you can try different target replacement strategies
    var = 3



    actor = Actor(sess, a_dim, LR_A, REPLACEMENT)
    critic = Critic(sess, s_dim, a_dim, LR_C, GAMMA, REPLACEMENT, actor.a, actor.a_)
    actor.add_grad_to_graph(critic.a_grads)

    sess.run(tf.global_variables_initializer())

    M = Memory(MEMORY_CAPACITY, dims=2 * s_dim + 2)

    mean_delay_time = []  # 记录延时
    mean_drop_item = []  # 记录阻塞的item
    mean_all_tf = []  # 记录tf
    mean_delay_num = [] #记录延迟item个数
    for episode in range(1000):
        step = 0

        print("迭代：", episode)
        print("_______________________________")
        clock = 0  # 系统时钟
        item_idx = 0  # item索引
        delay_time = []  # 记录延时
        drop_item = []  # 记录阻塞的item
        all_tf = []  # 记录tf
        while item_idx < len(item):
            if clock != item[item_idx][0]:  # 当前没有item到来
                clock += 1  # 更新时钟
                env.update_clock()  # 更新链路资源
            else:  # 有item到来
                it = item[item_idx]
                item_idx += 1
                ta = it[0]  # 到达时间
                td = it[1]  # 截止时间
                sn = str(it[2])  # 源点
                tn = str(it[3])  # 宿点
                data = it[4]  # 数据量

                print("item编号：", item_idx - 1)
                print("到达时间：", ta)
                print("截至时间：", ta + td)
                print("源点：", sn)
                print("宿点：", tn)
                print("数据量：", data)

                diff_path_ts_item_time_max_tf_block = []
                for path_k in range(3):  # 遍历源点到宿点的路径
                    path = Candidate_Paths[sn][tn][path_k]
                    diff_ts_item_time_max_tf_block = []
                    for ts in range(ta, ta + td):  # 遍历item开始时间
                        #print("开始时间：",ts)
                        diff_item_time_max_tf_block = []  # 不同持续时间的最大时间频谱连续度
                        for item_time in range(1, ta + td - ts + 1):  # 遍历item持续时间
                            #print("item_time:",item_time)
                            item_frequency = math.ceil(data / (12.5 * item_time)) + 1

                                # 获取item需要的频谱槽数量
                                #print("item_frequency:",item_frequency)
                            path_source = env.path_source(path)  # 获取路径资源矩阵
                                # 合成环境state
                            state = []
                            for i in range(len(path_source)):
                                for j in range(len(path_source[0])):
                                    state.append(path_source[i][j])
                            item_state = np.ones((env.height, env.width), dtype=np.int)
                            for i in range(item_time):
                                for j in range(item_frequency):
                                    item_state[i][j] = 0
                            for i in range(len(item_state)):
                                for j in range(len(item_state[0])):
                                    state.append(item_state[i][j])

                            state = np.array(state)
                                # 基于state选择action
                            action = actor.choose_action(state,var)
                            #print(len(action))
                            #action = int(np.clip(np.random.normal(action, var), 0, 14))
                            #print(action)


                            # 更新箱子
                            reward, path_source_, = env.update(path_source,action, item_frequency, item_time, ts-clock, td)

                            if reward > 0:
                                max_tf_current_item_time_block = [reward,action, item_time, item_frequency]
                                diff_item_time_max_tf_block.append(max_tf_current_item_time_block)
                                #print("采取的action:{},获得的奖励：{}".format(action,reward))
                                # print("___________________________________________________")
                                # 下一个state
                            state_ = []
                            for i in range(len(path_source_)):
                                for j in range(len(path_source_[0])):
                                    state_.append(path_source_[i][j])
                            for i in range(len(item_state)):
                                for j in range(len(item_state[0])):
                                    state_.append(item_state[i][j])
                            state_ = np.array(state_)
                                # 保存进记忆库
                            M.store_transition(state, action, reward, state_)



                            #if M.pointer > MEMORY_CAPACITY:
                            if (step >= 500) and (step % 100 == 0):
                                var *= .9995  # decay the action randomness
                                b_M = M.sample(BATCH_SIZE)
                                b_s = b_M[:, :s_dim]
                                b_a = b_M[:, s_dim: s_dim + a_dim]
                                b_r = b_M[:, -s_dim - 1: -s_dim]
                                b_s_ = b_M[:, -s_dim:]

                                critic.learn(b_s, b_a, b_r, b_s_)
                                actor.learn(b_s)
                        step += 1
                        diff_item_time_max_tf_block.sort()
                        if diff_item_time_max_tf_block:
                            max_tf_item_time_block = diff_item_time_max_tf_block[-1]
                            max_tf_item_time_block.append(ts)
                            diff_ts_item_time_max_tf_block.append(max_tf_item_time_block)
                    diff_ts_item_time_max_tf_block.sort()
                    if diff_ts_item_time_max_tf_block:
                        # print(diff_ts_item_time_max_tf_block)
                        max_tf_ts_item_time_block = diff_ts_item_time_max_tf_block[-1]
                        max_tf_ts_item_time_block.append(path_k)
                        diff_path_ts_item_time_max_tf_block.append(max_tf_ts_item_time_block)
                diff_path_ts_item_time_max_tf_block.sort()
                if diff_path_ts_item_time_max_tf_block:
                    max_tf_path_ts_item_time_block = diff_path_ts_item_time_max_tf_block[-1]  # [tf,action位置,持续时间，频谱槽，开始时间，路径编号]
                    max_tf = max_tf_path_ts_item_time_block[0]
                    select_position = max_tf_path_ts_item_time_block[1]
                    select_item_time = max_tf_path_ts_item_time_block[2]
                    select_item_frequency = max_tf_path_ts_item_time_block[3]
                    select_ts = max_tf_path_ts_item_time_block[4]
                    select_path = Candidate_Paths[sn][tn][max_tf_path_ts_item_time_block[5]]
                    env.update_path(select_path, select_position, select_item_time, select_item_frequency)

                    print("******DDPG结果：")
                    print("时间频谱效率为：", max_tf)
                    print("选择的空闲块为：", select_position)
                    print("持续时间为：", select_item_time)
                    print("频谱槽数量为：", select_item_frequency)
                    print("开始时间为：", select_ts)
                    print("选择路径为：", select_path)
                    print("_______________________________________________________________")

                    delay_time.append(select_ts - ta)
                    all_tf.append(max_tf)
                else:
                    drop_item.append(item_idx - 1)
                    #
                    print("item阻塞！！！！！！")
        mean_delay_time.append(mean(delay_time))
        mean_drop_item.append(len(drop_item))
        mean_all_tf.append(mean(all_tf))
        count = 0
        for i in delay_time:
            if i > 0:
                count += 1
        mean_delay_num.append(count / len(item))
        dict_bp = {'delay_time': mean_delay_time[-1], 'drop_item': mean_drop_item[-1]/50,
                   'all_tf': mean_all_tf[-1],
                   'delay_num': mean_delay_num[-1]}
        df = pd.DataFrame(dict_bp, index=[0])
        df.to_csv("DDPG.csv", sep=',', header=None, mode='a', index=False)

    print(mean_delay_num)
    print(mean_delay_time)
    print(mean_drop_item)
    print(mean_all_tf)

    #show_delay_item(mean_delay_num)
    #show_delay_time(mean_delay_time)
    #show_drop_item(mean_drop_item)
    #show_mean_tf(mean_all_tf)
#.plot_cost()

def show_mean_tf(tf):
    plt.plot(np.arange(len(tf)), tf)
    plt.ylabel('mean_tf')
    plt.xlabel('training steps')
    plt.savefig("images/"+'平均频谱率.png')
    plt.close()


def show_delay_item(delay_mean_item):
    plt.plot(np.arange(len(delay_mean_item)), delay_mean_item)
    plt.ylabel('delay_mean_item')
    plt.xlabel('training steps')
    plt.savefig("images/" + '延迟业务占比.png')
    plt.close()


def show_delay_time(delay_mean_time):
    plt.plot(np.arange(len(delay_mean_time)), delay_mean_time)
    plt.ylabel('delay_mean_time')
    plt.xlabel('training steps')
    plt.savefig("images/"+'平均时延.png')
    plt.close()


def show_drop_item(drop_item):
    plt.plot(np.arange(len(drop_item)), drop_item)
    plt.ylabel('drop_item')
    plt.xlabel('training steps')
    plt.savefig("images/"+'阻塞率.png')
    plt.close()