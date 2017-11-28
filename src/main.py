from computations import decompositions
from data import DataHandler
import util
from computations import relevanceFeedback
from data import taskRunner as tr
import pandas as pd
import time

# DataHandler.vectors()
t = time.time()

tr.task1_2CP()



print('Query : ', time.time() -t)
