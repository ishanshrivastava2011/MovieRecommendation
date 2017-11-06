from computations import decompositions
from data import DataHandler
import util
import pandas as pd
import time

DataHandler.vectors()
t = time.time()
X = DataHandler.actor_similarity_tagVector(3892753)
print(sorted(X,key=lambda x:x[1],reverse=True))
print('Query : ', time.time() -t)
