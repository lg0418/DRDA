from environment import Environment
from RL import DeepQNetwork
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import time
def run_RL(item,Candidate_Paths):
    env = Environment(10,15,52)
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.00001,
                      reward_decay=0.9,
                      e_greedy=1,
                      replace_target_iter=300,
                      memory_size=3000,
                      # output_graph=True
                      )
    mean_delay_time = []  # 记录延时
    mean_drop_item = []  # 记录阻塞的item
    mean_all_tf = []  # 记录tf
    mean_delay_num = [] #记录延迟item个数
    t_start = time.time()
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
                            #state.append(item_time)
                            #state.append(item_frequency)
                            state = np.array(state)
                                # 基于state选择action
                            action = RL.choose_action(state)

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
                            #state_.append(item_time)
                            #state_.append(item_frequency)
                            state_ = np.array(state_)
                                # 保存进记忆库
                            RL.store_transition(state, action, reward, state_)

                            if (step > 100) and (step % 5 == 0):
                                RL.learn()
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

                    print("*******RL结果：")
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

    t_end = time.time()
    print("耗时",t_end-t_start)
    
    print(mean_delay_num)
    print(mean_delay_time)
    print(mean_drop_item)
    print(mean_all_tf)

    show_delay_item(mean_delay_num)
    show_delay_time(mean_delay_time)
    show_drop_item(mean_drop_item)
    show_mean_tf(mean_all_tf)
    RL.plot_cost()

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