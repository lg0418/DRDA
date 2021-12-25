import numpy as np
import copy

node_dict = {'a':0,'b':1,'c':2,'d':3,'e':4,'f':5,'g':6,'h':7,'i':8,'j':9,'k':10}
class Environment:
    def __init__(self,height,width,link_num):
        self.height=height
        self.width=width
        self.link=[]
        for i in range(link_num):
            self.link.append(np.ones((self.height, self.width), dtype=np.int))
        self.counter=0
        self.i_min=0
        self.i_max=0
        self.j_min=0
        self.j_max=0
        self.n_actions = self.width
        self.n_features= self.width*self.height*2

    #重置环境
    def reset(self):
        for box in self.link:
            for i in range(self.height):
                for j in range(self.width):
                    if box[i][j]==0:
                        box[i][j]=1

    def path_source(self,path):
        pathsource = np.ones((self.height, self.width), dtype=np.int)
        for i in path:
            pathsource = self.merge_source(pathsource,self.link[node_dict[i]])
        return pathsource

    def merge_source(self,matrix1,matrix2):
        for i in range(len(matrix1)):
            for j in range(len(matrix1[0])):
                matrix1[i][j] = matrix1[i][j] and matrix2[i][j]
        return matrix1

    def update(self,box,action,item_width,item_height,start,deadline):  #action是箱子放置位置的左上角序号
        reward = 0
        if box[start][action] == 0 or action + item_width > len(box[0]):
            reward = -10
        else:
            flag = True
            for m in range(start, start + item_height):
                for n in range(action, action + item_width):
                    if box[m][n] == 0:
                        reward = -10
                        flag = False
                        break
                if flag == False:
                    break
            if flag == True:
                for m in range(start, start + item_height):
                    for n in range(action, action + item_width):
                        box[m][n] = 0


                #compactness = self.counter / ((self.i_max - self.i_min + 1) * (self.j_max - self.j_min +1))
                #print(cluster_size)
                #print(compactness)
                reward = self.time_frequency(box,start,action)
        return reward,box

    def time_frequency(self,matrix,start,action):
        occupied_num=0
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == 1:
                    occupied_num += 1
        A = occupied_num / (len(matrix)*len(matrix[0]))
        time_block = 0
        time_slice = 0
        frequency_block = 0
        frequency_slice = 0
        for i in range(len(matrix)):
            counter = 0
            j = 0
            while j < len(matrix[0]):
                if matrix[i][j] == 1:
                    counter +=1
                else:
                    if counter >= 2:
                        time_slice += counter - 1
                    time_block += 1
                    counter = 0
                j += 1
            if counter >=2:
                time_slice += counter - 1
            if counter != 0 :
                time_block += 1

        for j in range(len(matrix[0])):
            counter = 0
            i = 0
            while i < len(matrix):
                if matrix[i][j] == 1:
                    counter +=1
                else:
                    if counter >= 2:
                        frequency_slice += counter - 1
                    frequency_block += 1
                    counter = 0
                i += 1
            if counter >=2:
                frequency_slice += counter - 1
            if counter != 0:
                frequency_block += 1

        B = (time_slice + frequency_slice) / (time_block + frequency_block)
        temp = copy.deepcopy(matrix)
        self.counter = 0
        self.dfs(start, action, temp)
        cluster_size = self.counter
        C = cluster_size * A * B
        return C

    def free_block(self,item_time,item_frequency,matrix,start,deadline):
        freeblock=[]
        tmatrix = copy.deepcopy(matrix)
        st = []
        end = []

        if (start + item_time - 1) < deadline:

            counter = 0
            j = 0
            while j < len(matrix[0]):
                if matrix[start][j] == 1:
                    counter += 1
                else:
                    if counter >= item_frequency:
                        end.append(j)
                        st.append(j-counter)
                    counter = 0
                j += 1
            if counter >= item_frequency:
                end.append(j)
                st.append(j - counter)
            if len(st) == 0:
                return freeblock
            else:
                for idx in range(len(st)):
                    for n in range(st[idx],end[idx] - item_frequency):
                        flag = True
                        for m in range(start,start+item_time):
                            for k in range(n,n+item_frequency):
                                if tmatrix[m][k] == 0:
                                    flag = False
                                    break
                            if flag == False:
                                break
                        if flag == True:
                            freeblock.append(start * len(tmatrix[0]) + n)
                            break
                return freeblock

        return freeblock
    def occupy_free_block(self,matrix,position,item_time,item_frequency):
        x = position // len(matrix[0])
        y = position % len(matrix[0])
        for i in range(x,x+item_time):
            for j in range(y,y+item_frequency):
                matrix[i][j]=0
        return matrix

    def free_occupy(self,matrix,position,item_time,item_frequency):
        x = position // len(matrix[0])
        y = position % len(matrix[0])
        for i in range(x, x + item_time):
            for j in range(y, y + item_frequency):
                matrix[i][j] = 1
        return matrix

    def update_clock(self):
        for l in self.link:
            for i in range(0,len(l)):
                for j in range(0,len(l[0])):
                    if i == len(l)-1:
                        l[i][j] = 1
                    else:
                        l[i][j] = l[i+1][j]

    def update_path(self,path,position,item_time,item_frequency):
        for i in path:
            self.link[node_dict[i]] = self.occupy_free_block(self.link[node_dict[i]],position,item_time,item_frequency)


    def dfs(self,i,j,temp):

        if i < 0 or j < 0 or i >= len(temp) or j >= len(temp[0]) or temp[i][j] != 0:
            return
        temp[i][j] = 1
        self.counter += 1
        if i > self.i_max:
            self.i_max = i
        if i < self.i_min:
            self.i_min = i
        if j > self.j_max:
            self.j_max = j
        if j < self.j_min:
            self.j_min = j
        self.dfs(i + 1, j, temp)
        self.dfs(i - 1, j, temp)
        self.dfs(i, j + 1, temp)
        self.dfs(i, j - 1, temp)

    def render(self):
        #print("箱子：\n",self.box)
        return self.box

    def efficiency(self):
        empty=0
        for x in range(self.height):
            for y in range(self.width):
                if self.box[x][y] == 0:
                    empty+=1
        return (self.height*self.width-empty) / (self.height*self.width)

    def clear(self,action,width,height):
        i = action // self.width
        j = action % self.width
        for m in range(i, i + height):
            for n in range(j, j + width):
                self.box[m][n] = 0
    def find_freeblock(self,matrix,tes,W,item_time,item_frequency,pathk,hp):
        final_B = []
        final_M = float("inf")
        for t in range(tes,tes+W):
            for f in range(0,len(matrix[0])-item_frequency+1):
                B=[pathk,f,f+item_frequency,t,t+item_time]
                flag = True
                for i in range(B[3],B[4]):
                    for j in range(B[1],B[2]):
                        if matrix[i][j] == 0:
                            flag = False
                            break
                    if flag == False:
                        break
                if flag == True:
                    Rc = item_frequency*item_time*hp
                    Dup = f
                    Ddown = len(matrix[0])-(f+item_frequency)
                    Df = min(Dup,Ddown)
                    Dt = t - tes
                    Rd = 0
                    Rleft = 0
                    for j in range(B[1],B[2]):
                        if matrix[t-1][j] == 1:
                            Rleft += 1
                    if Dup < Ddown:
                        Rup = 0
                        if Dup != 0:
                            for i in range(t-1,t+item_time):
                                if matrix[i][f+item_frequency] == 1:
                                    Rup += 1
                        Rd = Rleft + Rup
                    if Dup > Ddown:
                        Rdown = 0
                        if Ddown != 0:
                            for i in range(t-1,t+item_time):
                                if matrix[i][f-1] == 1:
                                    Rdown += 1
                        Rd = Rleft + Rdown
                    if Dup == Ddown:
                        Rup = Rdown =0
                        if Dup != 0:
                            for i in range(t-1,t+item_time):
                                if matrix[i][f+item_frequency] == 1:
                                    Rup += 1
                            for i in range(t-1,t+item_time):
                                if matrix[i][f-1] == 1:
                                    Rdown += 1
                        Rd = Rleft + min(Rup,Rdown)
                    M = Rc + Df +  3* Dt + Rd

                    if M < final_M:
                        final_M = M
                        final_B.clear()
                        final_B.append(B[0])
                        final_B.append(B[1])
                        final_B.append(B[2])
                        final_B.append(B[3])
                        final_B.append(B[4])
        return final_M, final_B

if __name__ == '__main__':
    #env=Environment()
    #env.rest()
    #print(env.box)
    #reward,_state,done=env.update(10,10,2)
    item = [[2, 3],  # 第一个数是item的height，第二个是width
            [1, 1],
            [1, 1],
            [1, 4],
            [2, 3],
            [1, 1],
            [1, 1]]
    it=item[0]
    #print(it[0],it[1])
    num = 0
    i=0
    while num < 10:
        a = np.random.poisson(10,1)
        print(a)
        if (a[0] != 0):
            arrive = i
            leave = int(
                round((i + np.random.exponential(10,1)[0]),0)
            )
            print(round((i + np.random.exponential(10,1)[0]),0))
            num +=1
        i +=1

