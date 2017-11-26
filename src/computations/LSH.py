#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 19:02:48 2017

@author: ishanshrivastava
"""

from computations import decompositions
from data import DataHandler
import pandas as pd
from collections import defaultdict
from operator import itemgetter
from util import constants
from util import formatter
import numpy as np
from computations import metrics
import random
LHashTables = defaultdict(lambda: defaultdict(list))

def createAHashFunction(d,w):
    r = [random.gauss(0,1) for i in range(d)]
    b = random.uniform(0,w)
    hashFunction = tuple((r,b))
    return hashFunction

#k is the number of hashes per layer
def createAHashFamily(k,d,w):
    hashFunctions = list()
    for kth in range(k):
        hashFunctions.append(createAHashFunction(d,w))
    return hashFunctions

#L is the number of Layers
def createLayers(L,k,d,w):
    layerTables = list()
    for layer in range(L):
        layerTables.append(createAHashFamily(k,d,w))
    return layerTables
        
def getHashKeyForAHashFunction(hashFunction,point,w):
    r = hashFunction[0]
    b = hashFunction[1]
    return str(int((np.dot(r,point.T)+b)/w))

def getHashKeyForAHashFamily(hashFunctions,point,w):
    return "".join([getHashKeyForAHashFunction(hashFunction,point,w) for hashFunction in hashFunctions])

def mapPointIndexToLBuckets(index,L,w,layerTables,MoviesinLatentSpace_SVD_Matrix):
    global LHashTables
    point = MoviesinLatentSpace_SVD_Matrix[index]
    for layer in range(L):
        hashFunctions = layerTables[layer]
        key = getHashKeyForAHashFamily(hashFunctions,point,w)
        LHashTables[layer][key].append(index)
#    print(str(index)+' Done')

def createAndGetLSH_IndexStructure(L,k,d,w,MoviesinLatentSpace_SVD_Matrix):
    global LHashTables
    layerTables = createLayers(L,k,d,w)        
    for index in range(MoviesinLatentSpace_SVD_Matrix.shape[0]):
        mapPointIndexToLBuckets(index,L,w,layerTables,MoviesinLatentSpace_SVD_Matrix)
    return layerTables,LHashTables
