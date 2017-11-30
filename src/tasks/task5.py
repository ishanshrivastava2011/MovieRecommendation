# -*- coding: utf-8 -*-

from computations import tasksBusiness

def task5(method):
    if(method=="RNN"):
        tasksBusiness.task5_1()
    elif(method=="DT"):
        tasksBusiness.task5_2()
    elif(method=="SVM"):
        tasksBusiness.task5_3()