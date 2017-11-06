import numpy as np
from util import constants
from sklearn.cluster import KMeans

"""
Multiplies the array by an identity matrix of the same .
Expects inputArray as a list.
Intention is to convert the sigma function to a matrix.
"""
def convertArrayToMatrix(inputArray):
    return np.matrix(inputArray) * np.matrix(np.eye(len(inputArray)))

def normalizer(min, max, value):
	return ((value - min + 0.0001) / (max - min + 0.0001))*constants.NORMALIZATION_VALUE

def coordinate_bounder(X,MN,STD,std_count):
	size = len(X)
	for i in range(0,size):
		if X[i]>MN[i]+(std_count*STD[i]) or X[i]<MN[i]-(std_count*STD[i]):
			return False
	return True

def dist_from_origin(X):
	return sum(list(map(lambda x:x**2,X)))**0.5

def outlier_removalx(data,std_count):
	distances = list(map(lambda x: dist_from_origin(x),data))
	mean = np.mean(distances)
	stdev = np.std(distances)
	filter_values = []
	for point in data:
		dist = dist_from_origin(point)
		if dist<mean+(std_count*stdev) and dist>mean-(std_count*stdev):
			filter_values.append(point)

	return np.array(filter_values)


def kmeans_outlier_removal_iterative(data,numgroups):
	original_size = data.__len__()
	model = None
	prev_size = original_size
	while data.__len__() >= original_size * .75:
		model = outlier_removalkm(data,numgroups)
		if model[2]:
			data = model[0]
		else:
			break
		if prev_size == data.__len__():
			break
		prev_size = data.__len__()
		#print(data.__len__())
	return data

def outlier_removalkm(data,numgroups):
	size = data.__len__()
	reduced = False
	kmean = KMeans(n_clusters=numgroups).fit(data)
	labels = kmean.labels_
	clusters = []
	for i in range(numgroups):
		clusters.append(list())

	for i in range(size):
		for j in range(numgroups):
			if labels[i] == j:
				clusters[j].append(data[i])

	for clus in clusters:
		if clus.__len__() > size*.75:
			data = clus
			reduced = True
			break

	return [data,kmean,reduced]

def splitGroup(data, numgroups):
	groupArray = []
	for i in range(numgroups):
		groupArray.append(list())
	newData = kmeans_outlier_removal_iterative(data,3)
	kmean = KMeans(n_clusters=numgroups).fit(newData)
	labels = kmean.predict(data)
	for i in range(data.__len__()):
		groupArray[labels[i]].append(i)

	return groupArray

def outlier_removal(data,std_count):
	axes = list(zip(*data))
	means = list(map(lambda x:np.mean(x),axes))
	stdevs = list(map(lambda x:np.std(x),axes))
	return list(filter(lambda x:coordinate_bounder(x,means,stdevs,std_count),data))



