# -*- coding: utf-8 -*-

from data import taskRunner

def task1_2(method):
    if(method=="SVD"):
        taskRunner.task1_2SVD()
    elif(method=="PCA"):
        taskRunner.task1_2PCA()
    elif(method=="LDA"):
        taskRunner.task1_2LDA()
    elif(method=="CPD"):
        taskRunner.task1_2CP()
    elif(method=="PPR"):
        taskRunner.task1_2PageRank()
    elif(method=="COM"):
        taskRunner.task1_2Combined()
    else:
        print("Method used is wrong. Please use SVD/PCA/LDA/CPD/PPR/COM")