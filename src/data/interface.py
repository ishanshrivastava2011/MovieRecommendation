# -*- coding: utf-8 -*-

import sys
from tasks import task1_2, task3_4, task5

task_dict = {'task1_2': task1_2.task1_2, 'task3_4': task3_4.task3_4, 'task5': task5.task5}

if __name__ == '__main__':
    directive = sys.argv[1]
    if directive == 'task1_2':
        arg1 = sys.argv[2]
        task_dict[directive](arg1)
    elif directive == 'task3_4':
        task_dict[directive]()
    elif directive == 'task5':
        arg1 = sys.argv[2]
        task_dict[directive](arg1)
    else:
        print("Invalid task. Goodbye")