# -*- coding: utf-8 -*-
from collections import defaultdict
import numpy as np
from operator import itemgetter
import networkx as nx

def PPR(similaritDataframe,seed,alpha):
    A=np.matrix(similaritDataframe)
    G=nx.from_numpy_matrix(A)
    prob = 1/len(seed)
    actors = list(similaritDataframe.index)
    seed = [check(i,seed,actors) for i in seed]
    seed = list(filter(None.__ne__, seed))
    personalization_map = defaultdict()
    for i in range(0,len(similaritDataframe)):
        if i in seed:
            personalization_map[i]=prob
        else:
            personalization_map[i]=0
    pr = nx.pagerank(G, alpha=alpha,personalization=personalization_map)  
    return sorted(pr.items(),key = itemgetter(1),reverse=True)[0:11]

#seed = {17838,61523}

def check(i,seed,actors):
    if i in actors:
        return actors.index(i)
    else:
        return []