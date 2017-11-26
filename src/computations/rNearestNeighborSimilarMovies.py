#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 23:30:27 2017

@author: ishanshrivastava
"""
import pandas as pd
import numpy as np
from collections import defaultdict
from operator import itemgetter
from util import constants
import time
from computations import LSH as lsh
import itertools
from sklearn import metrics
from computations import relevanceFeedback

def getMoviesInLDifferentHashBuckets(L,point,w,layerTables,LHashTables):
    movieIndices = list()
    for layer in range(L):
        hashFunctions = layerTables[layer]
        key = lsh.getHashKeyForAHashFamily(hashFunctions,point,w)
        movieIndices.append(LHashTables[layer][key])
    return movieIndices


#movieid = moviesList[4]
#r = 10
def getRNearestNeighbors(movieid,r,MoviesinLatentSpace,layerTables,LHashTables_result):
    t1 = time.time()    
    L = len(layerTables)
    w = constants.W
#    MoviesinLatentSpace = pd.read_csv(constants.DIRECTORY+'MoviesinLatentSpace_SVD_MDS.csv',index_col = 0)
    MoviesinLatentSpace_Matrix = np.matrix(MoviesinLatentSpace,dtype = np.float32)
#    layerTables,LHashTables_result = lsh.createAndGetLSH_IndexStructure(L,k,d,w,MoviesinLatentSpace_Matrix)
    
    moviesList =list( MoviesinLatentSpace.index)
    givenMovieidIndex = moviesList.index(movieid)
    point = MoviesinLatentSpace_Matrix[givenMovieidIndex].astype(np.float32)

    nearbyMovieIndices = list(itertools.chain.from_iterable(getMoviesInLDifferentHashBuckets(L,point,w,layerTables,LHashTables_result)))
    uniqueNearbyMovieIndices = list(set(nearbyMovieIndices))
    
    print('Number of Unique movies considered: '+str(len(uniqueNearbyMovieIndices)))
    print('Overall number of movies considered: '+str(len(nearbyMovieIndices)))
    
    if givenMovieidIndex in uniqueNearbyMovieIndices:
        uniqueNearbyMovieIndices = list(set(nearbyMovieIndices)-set([givenMovieidIndex]))
        nearbyMovieList = [moviesList[i ] for i in uniqueNearbyMovieIndices]
        MoviesinLatentSpace_SVD_Matrix_subset = MoviesinLatentSpace_Matrix[uniqueNearbyMovieIndices]
        distances = relevanceFeedback.euclideanMatrixVector(MoviesinLatentSpace_SVD_Matrix_subset,point)
        
        nearestMovieIndices = np.argsort(distances[0])[:r]
        nearestMovies = [nearbyMovieList[i] for i in np.array(nearestMovieIndices)[0]][:r]
    
    
    #Brute Forece method for Finding similarities
    moviesListTest = moviesList
    moviesListTest.remove(movieid)
    allButGivenMovieList = list(set(range(MoviesinLatentSpace_Matrix.shape[0]))-set([givenMovieidIndex]))
    distancesTest = relevanceFeedback.euclideanMatrixVector(MoviesinLatentSpace_Matrix[allButGivenMovieList],point)
    nearestMovieIndicesTest = np.argsort(distancesTest[0])
    nearestMoviesTest = [moviesListTest[i] for i in np.array(nearestMovieIndicesTest)[0]][:r]
    
    print(len(set(nearestMovies).intersection(set(nearestMoviesTest[:len(nearestMovies)]))))
#    print(nearestMoviesTest)
#    print(nearestMovies)
    print(time.time()-t1)
    return nearestMovies,nearestMoviesTest
    
