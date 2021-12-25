from environment import Environment
from numpy import *
import numpy as np
#item=[到达时间，截止时间、源点、宿点、数据量]


def RDDR(item,Candidate_Paths):
    env = Environment(10, 15, 44)
    clock = 0    #系统时钟
    item_idx = 0  #item索引
    delay_time = [] #记录延时
    drop_item = []  #记录阻塞的item
    all_tf = [] #记录tf
    while item_idx < len(item):
        if clock != item[item_idx][0]:    #当前没有item到来
            clock +=1       #更新时钟
            env.update_clock()  #更新链路资源
        else:    #有item到来
            it = item[item_idx]
            item_idx += 1
            ta = it[0]    #到达时间
            td = it[1]    #截止时间
            sn = str(it[2])   #源点
            tn = str(it[3])   #宿点
            data = it[4]      #数据量

            print("item编号：",item_idx-1)
            print("到达时间：",ta)
            print("截至时间：", ta+td)
            print("源点：", sn)
            print("宿点：", tn)
            print("数据量：", data)

            final_B=[]
            final_M=float("inf")
            tes = np.random.randint(ta, ta + 2)
            W = np.random.randint(4, 7)
            item_time = np.random.randint(1,3)  # 遍历item持续时间
            item_frequency = math.ceil(data / (12.5 * item_time)) + 1
            for path_k in range(3):   #遍历源点到宿点的路径
                path = Candidate_Paths[sn][tn][path_k]
                path_source = env.path_source(path)   #获取路径资源矩阵
                M,B = env.find_freeblock(path_source,tes-clock,W,item_time,item_frequency,path_k,len(path)-1)  #获取满足item时间片和频谱槽的空闲块
                if M < final_M:
                    final_M = M
                    final_B.clear()
                    final_B.append(B[0])
                    final_B.append(B[1])
                    final_B.append(B[2])
                    final_B.append(B[3])
                    final_B.append(B[4])
            if final_B:
                p = Candidate_Paths[sn][tn][final_B[0]]
                env.update_path(p,final_B[3]*env.width+final_B[1],item_time,item_frequency)
                p_source = env.path_source(p)
                tf = env.time_frequency(p_source,final_B[3],final_B[1])

                print("*******Min RDDR算法结果：")
                print("时间频谱效率为：",tf)
                print("选择的空闲块为：",final_B[3]*env.width+final_B[1])
                print("持续时间为：",item_time)
                print("频谱槽数量为：", item_frequency)
                print("开始时间为：", ta+final_B[3])
                print("选择路径为：", Candidate_Paths[sn][tn][final_B[0]])
                print("_______________________________________________________________")

                delay_time.append(final_B[3]-ta)
                all_tf.append(tf)
            else:
                drop_item.append(item_idx-1)
                print("item阻塞！！！！！！")
                print("持续时间为：", item_time)
                print("频谱槽数量为：", item_frequency)
                print("_______________________________________________________________")
    print("RDDR")
    print("容量阻塞率：", len(drop_item) / len(item))
    count = 0
    for i in delay_time:
        if i > 0:
            count += 1
    print("产生初始时延的业务占比：", count / len(item))
    print("平均初始时延:", mean(delay_time))
    print("平均时间频谱效率：", mean(all_tf))
    print(len(all_tf))
    file_handle = open('1.txt', mode='a')
    file_handle.write('LTC \n容量阻塞率：')
    file_handle.write(str(len(drop_item) / len(item)))
    file_handle.write('\n 产生初始时延的业务占比：')
    file_handle.write(str(count / len(item)))
    file_handle.write('\n 平均初始时延:')
    file_handle.write(str(mean(delay_time)))
    file_handle.write('\n 平均时间频谱效率：')
    file_handle.write(str(mean(all_tf)))
    file_handle.write('\n')
    file_handle.close()



