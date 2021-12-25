from environment import Environment
from numpy import *
#item=[到达时间，截止时间、源点、宿点、数据量]


def LSR(item,Candidate_Paths):
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

            diff_path_ts_item_time_max_tf_block=[]
            for path_k in range(3):   #遍历源点到宿点的路径
                path = Candidate_Paths[sn][tn][path_k]
                diff_ts_item_time_max_tf_block = []
                for ts in range(ta,ta+td):  #遍历item开始时间
                    #print("开始时间：",ts)
                    diff_item_time_max_tf_block = []  #不同持续时间的最大时间频谱连续度
                    for item_time in range(1,ta+td-ts+1):  #遍历item持续时间
                        #print("item_time:",item_time)
                        item_frequency = math.ceil(data / (12.5 * item_time)) + 1
                        #获取item需要的频谱槽数量
                        #print("item_frequency:",item_frequency)
                        path_source = env.path_source(path)   #获取路径资源矩阵
                        freeblock = env.free_block(item_time,item_frequency,path_source,ts-clock,td)  #获取满足item时间片和频谱槽的空闲块
                        if freeblock:
                            tf = []         #记录不同的空闲块产生的时间频谱连续度
                            for block in freeblock:  #遍历不同的空闲块
                                path_source = env.occupy_free_block(path_source,block,item_time,item_frequency)  #占用该空闲块
                                item_tf = env.time_frequency(path_source,block//len(path_source[0]),block%len(path_source[0]))   #获取占用该空闲块时产生的时间频谱连续度
                                tf.append(item_tf)
                                path_source = env.free_occupy(path_source,block,item_time,item_frequency)   #释放该空闲块
                            max_tf_block = freeblock[tf.index(max(tf))]  #获取产生最大时间频谱连续度的空闲块
                            max_tf_current_item_time_block = [item_frequency,max(tf),max_tf_block,item_time]
                            diff_item_time_max_tf_block.append(max_tf_current_item_time_block)
                            #print(diff_item_time_max_tf_block)

                    diff_item_time_max_tf_block = sorted(diff_item_time_max_tf_block,key=lambda x: (x[0], -x[1]))
                    #print(diff_item_time_max_tf_block)
                    if diff_item_time_max_tf_block:
                        max_tf_item_time_block = diff_item_time_max_tf_block[0]
                        max_tf_item_time_block.append(ts)
                        diff_ts_item_time_max_tf_block.append(max_tf_item_time_block)
                        #print(diff_ts_item_time_max_tf_block)
                diff_ts_item_time_max_tf_block = sorted(diff_ts_item_time_max_tf_block,key=lambda x: (x[0], -x[1]))
                #print(diff_ts_item_time_max_tf_block)
                if diff_ts_item_time_max_tf_block:
                    max_tf_ts_item_time_block = diff_ts_item_time_max_tf_block[0]
                    max_tf_ts_item_time_block.append(path_k)
                    diff_path_ts_item_time_max_tf_block.append( max_tf_ts_item_time_block)
                    #print(diff_path_ts_item_time_max_tf_block)
            diff_path_ts_item_time_max_tf_block = sorted(diff_path_ts_item_time_max_tf_block,key=lambda x: (x[0], -x[1]))
            #print(diff_path_ts_item_time_max_tf_block)
            if diff_path_ts_item_time_max_tf_block:

                max_tf_path_ts_item_time_block = diff_path_ts_item_time_max_tf_block[0]
                select_item_frequency = max_tf_path_ts_item_time_block[0]
                max_tf = max_tf_path_ts_item_time_block[1]
                select_position = max_tf_path_ts_item_time_block[2]
                select_item_time = max_tf_path_ts_item_time_block[3]
                select_ts = max_tf_path_ts_item_time_block[4]
                select_path = Candidate_Paths[sn][tn][max_tf_path_ts_item_time_block[5]]
                env.update_path(select_path,select_position,select_item_time,select_item_frequency)

                print("*******LSR算法结果：")
                print("时间频谱效率为：",max_tf)
                print("选择的空闲块为：",select_position)
                print("持续时间为：",select_item_time)
                print("频谱槽数量为：", select_item_frequency)
                print("开始时间为：", select_ts)
                print("选择路径为：", select_path)
                print("_______________________________________________________________")

                delay_time.append(select_ts-ta)
                all_tf.append(max_tf)
            else:
                drop_item.append(item_idx-1)
                print("item阻塞！！！！！！")
                print("_______________________________________________________________")
    print("LSR")
    print("容量阻塞率：",len(drop_item)/len(item))
    count = 0
    for i in delay_time:
        if i > 0:
            count += 1
    print("产生初始时延的业务占比：",count / len(item))
    print("平均初始时延:",mean(delay_time))
    print("平均时间频谱效率：",mean(all_tf))




