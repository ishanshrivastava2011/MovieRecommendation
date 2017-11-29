#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 22:51:57 2017

@author: ishanshrivastava
"""

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from itertools import chain

def euclideanDistance(sparseMatrix1,sparseMatrix2):
    return pairwise_distances(sparseMatrix1, sparseMatrix2)

def getAccuracy(predictions,test_label):
    return sum(np.array(predictions) == np.array(test_label))/float(len(predictions))

def getNNfor1Test(test,trainMatrix):
    neighbors = euclideanDistance(trainMatrix,test)
    return neighbors

def NN(trainMatrix,testMatrix):
    KNNForAllTest = [getNNfor1Test(test,trainMatrix) for test in testMatrix]
    return KNNForAllTest

def sortAndReturnLabel(maxKIndices,train_label):
    labels = [train_label[int(indexOfImage)] for indexOfImage in maxKIndices]
    return labels

def sortAllNNAndGetLabels(NNForAllTest,K,train_label):
#    maxK = max(Ks)
    maxKNNLabels = [sortAndReturnLabel(np.argsort(list(chain(*NN)))[0:K],train_label) for NN in NNForAllTest]
    return maxKNNLabels

