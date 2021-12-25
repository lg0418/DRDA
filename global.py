import networkx as nx
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import heapq
import sys
import numpy as np
from numpy import *
from LSR import LSR
from LTC import LTC
from MPT import MPT
from environment import Environment
from run import run_RL
import _thread
from Min_RDDR import RDDR
from run_DDPG import run_DDPG

np.random.seed(1)
class nx_Graph:
    def __init__(self):
        self.vertices = {}

    def add_vertex(self, name, edges):
        self.vertices[name] = edges

    def get_shortest_path(self, startpoint, endpoint):

        distances = {}


        previous = {}


        nodes = []


        for vertex in self.vertices:
            if vertex == startpoint:

                distances[vertex] = 0
                heapq.heappush(nodes, [0, vertex])
            elif vertex in self.vertices[startpoint]:

                distances[vertex] = self.vertices[startpoint][vertex]
                heapq.heappush(nodes, [self.vertices[startpoint][vertex], vertex])
                previous[vertex] = startpoint
            else:

                distances[vertex] = sys.maxsize
                heapq.heappush(nodes, [sys.maxsize, vertex])
                previous[vertex] = None

        while nodes:

            smallest = heapq.heappop(nodes)[1]
            if smallest == endpoint:
                shortest_path = []
                lenPath = distances[smallest]
                temp = smallest
                while temp != startpoint:
                    shortest_path.append(temp)
                    temp = previous[temp]

                shortest_path.append(temp)
            if distances[smallest] == sys.maxsize:

                break

            for neighbor in self.vertices[smallest]:
                dis = distances[smallest] + self.vertices[smallest][neighbor]
                if dis < distances[neighbor]:
                    distances[neighbor] = dis

                    previous[neighbor] = smallest
                    for node in nodes:
                        if node[1] == neighbor:

                            node[0] = dis
                            break
                    heapq.heapify(nodes)
        return distances, shortest_path, lenPath

    def getMinDistancesIncrement(self, inputList):
        inputList.sort()
        lenList = [v[0] for v in inputList]
        minValue = min(lenList)
        minValue_index = lenList.index(minValue)
        minPath = [v[1] for v in inputList][minValue_index]
        return minValue, minPath, minValue_index

    def k_shortest_paths(self, start, finish, k=3):

        distances, _, shortestPathLen = self.get_shortest_path(start, finish)
        num_shortest_path = 0
        paths = dict()
        distancesIncrementList = [[0, finish]]
        while num_shortest_path < k:
            path = []
            # distancesIncrementList = self.deleteCirclesWithEndpoint(distancesIncrementList,finish)
            minValue, minPath, minIndex = self.getMinDistancesIncrement(distancesIncrementList)
            smallest_vertex = minPath[-1]
            distancesIncrementList.pop(minIndex)

            if smallest_vertex == start:
                path.append(minPath[::-1])
                num_shortest_path += 1

                paths[path[0]] = minValue + shortestPathLen

                continue

            for neighbor in self.vertices[smallest_vertex]:
                incrementValue = minPath
                increment = 0
                if neighbor == finish:

                    continue
                if distances[smallest_vertex] == (distances[neighbor] + self.vertices[smallest_vertex][neighbor]):
                    increment = minValue
                elif distances[smallest_vertex] < (distances[neighbor] + self.vertices[smallest_vertex][neighbor]):
                    increment = minValue + distances[neighbor] + self.vertices[smallest_vertex][neighbor] - distances[
                        smallest_vertex]
                elif distances[neighbor] == (distances[smallest_vertex] + self.vertices[smallest_vertex][neighbor]):
                    increment = minValue + 2 * self.vertices[smallest_vertex][neighbor]
                distancesIncrementList.append([increment, incrementValue + neighbor])
        return paths

def create_item(size,lamda,u):

    item=[]
    arrival_time = []
    t = 0
    num = 0
    deadline_time = np.random.randint(6,10,size)
    while num < size:
        a = np.random.poisson(lamda, 1)
        if (a[0] != 0):

            arrival_time.append(int(t))
            num += 1
        t += np.random.exponential( u, 1)



    node_dict = {'0': 'a', '1': 'b', '2': 'c', '3': 'd', '4': 'e', '5': 'f', '6': 'g', '7': 'h', '8': 'i', '9': 'j', '10': 'k'}

    st = np.random.randint(0, 11, size=size)
    source_node = []
    for i in st:
        source_node.append(node_dict[str(i)])
    target_node = []
    data = []

    for i in st:
        t = np.random.randint(0, 11)
        while True:
            if i == t:
                t = np.random.randint(0, 11)
            else:
                break
        target_node.append(node_dict[str(t)])

    for i in range(size):
        t =  np.random.randint(0, 3)
        if t == 0:
            data.append(25)
        if t == 1:
            data.append(50)
        if t ==2:
            data.append(100)



    for i in range(size):
        temp=[]
        temp.append(arrival_time[i])
        temp.append(deadline_time[i])
        temp.append(source_node[i])
        temp.append(target_node[i])
        temp.append(data[i])
        item.append(temp)
    item.sort()
    return item

if __name__ == '__main__':


    g = nx_Graph()
    g.add_vertex('a', {'b': 1310, 'c': 760, 'd': 390, 'g':740 })
    g.add_vertex('b', {'a': 1310, 'c': 550, 'e': 390, 'h': 450})
    g.add_vertex('c', {'a': 760, 'b': 550, 'd': 660, 'e': 210, 'f': 390})
    g.add_vertex('d', {'a': 390, 'c': 660, 'g': 340, 'h': 1090, 'j': 660})
    g.add_vertex('e', {'b': 390, 'c': 210, 'f': 220, 'h': 300, 'k': 930})
    g.add_vertex('f', {'c': 390, 'e': 220, 'g': 730,'h':400, 'i': 350})
    g.add_vertex('g', {'a': 740, 'd': 340, 'f': 730, 'i': 565, 'j': 320})
    g.add_vertex('h', {'b': 450, 'd': 1090, 'e': 300,'f': 400, 'i': 600, 'k': 820})
    g.add_vertex('i', {'f': 350, 'g': 565, 'h':600, 'j': 730, 'k': 320})
    g.add_vertex('j', {'d': 660,'g': 320, 'i': 730, 'k': 820})
    g.add_vertex('k', {'e': 930, 'h': 820, 'i': 320, 'j': 820})
    k = 3

    node_list = ['a','b', 'c', 'd', 'e', 'f','g','h','i','j','k']
    Candidate_Paths = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))

    for s in node_list:
        for d in node_list:
            if s != d:
                paths = g.k_shortest_paths(s, d, k)
                #print(s, d, paths.keys())
                for k_id, path in enumerate(paths.keys()):
                    #print("k_id:",k_id)
                    #print("path:",path)
                    path_list = []
                    for p in path:
                        path_list.append(str(p))
                    Candidate_Paths[s][d][k_id] = path_list
                    #print(Candidate_Paths[s][d][k_id])
                    #print(path_list)
                    #print("----------------------")
    #print(Candidate_Paths['1']['2'][0])
    #sys.stdout = open('recode.log', mode = 'w',encoding='utf-8')



    lamda = 10

    for u in [0.04]:

        item = create_item(50,lamda,u)

        dur = []
        for i in item:
            dur.append(i[1]-i[0])




        run_DDPG(item, Candidate_Paths)
        #run_RL(item, Candidate_Paths)
        #LTC(item, Candidate_Paths)
        #LSR(item, Candidate_Paths)
        #RDDR(item, Candidate_Paths)













