# -*- coding: utf-8 -*-

import sys
from tasks import task1a
from tasks import task1b
from tasks import task1c
from tasks import task1d
from tasks import task2
from tasks import task3
from tasks import task4

task_dict = {('task1','a'): task1a.task1a, ('task1','b'): task1b.task1b, ('task1','c'): task1c.task1c, ('task1','d'): task1d.task1d,
			 ('task2','a'):task2.task2a, ('task2','b'):task2.task2b, ('task2','c'):task2.task2c, ('task2','d'):task2.task2d,
			 ('task3', 'a'): task3.task3a, ('task3', 'b'): task3.task3b,
			 ('task4', ''): task4.task4}

if __name__ == '__main__':
	directive = sys.argv[1]
	if not directive == 'task4':
		sub_directive = sys.argv[2]
		if directive == 'task1':
			arg1 = sys.argv[3]
			arg2 = sys.argv[4]
			if sub_directive in ['c', 'd']:
				arg1 = int(arg1)
			task_dict[(directive,sub_directive)](arg1,arg2)
		elif directive == 'task2':
			if sub_directive in ['a','b']:
				task_dict[(directive, sub_directive)]()
			else:
				arg1 = sys.argv[3]
				task_dict[(directive, sub_directive)](arg1)
		elif directive == 'task3':
			arg1 = sys.argv[3]
			seedList = [int(k) for k in arg1.split(",")]
			task_dict[(directive, sub_directive)](seedList)
	else:
		arg1 = int(sys.argv[2])
		task_dict[(directive, '')](arg1)