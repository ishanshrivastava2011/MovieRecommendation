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
def getRNearestNeighbors(movieid,moviePoint,r,MoviesinLatentSpace,layerTables,LHashTables_result):
    L = len(layerTables)
    w = constants.W
    MoviesinLatentSpace_Matrix = np.matrix(MoviesinLatentSpace,dtype = np.float32)
    moviesList =list( MoviesinLatentSpace.index)
    givenMovieidIndex = moviesList.index(movieid)
    nearbyMovieIndices = list(itertools.chain.from_iterable(getMoviesInLDifferentHashBuckets(L,moviePoint,w,layerTables,LHashTables_result)))
    uniqueNearbyMovieIndices = list(set(nearbyMovieIndices))
    
    print('\nNumber of Unique movies considered: '+str(len(uniqueNearbyMovieIndices)))
    print('Overall number of movies considered: '+str(len(nearbyMovieIndices))+"\n")
    
    uniqueNearbyMovieIndices = list(set(nearbyMovieIndices)-set([givenMovieidIndex]))
    nearbyMovieList = [moviesList[i ] for i in uniqueNearbyMovieIndices]
    MoviesinLatentSpace_SVD_Matrix_subset = MoviesinLatentSpace_Matrix[uniqueNearbyMovieIndices]
    distances = relevanceFeedback.euclideanMatrixVector(MoviesinLatentSpace_SVD_Matrix_subset,moviePoint)
    nearestMoviesDistance = np.sort(distances[0])[:r]
    nearestMovieIndices = np.argsort(distances[0])[:r]
    nearestMovies = [nearbyMovieList[i] for i in np.array(nearestMovieIndices)[0]][:r]
    
    
    #Brute Forece method for Finding similarities
    moviesListTest = moviesList
    moviesListTest.remove(movieid)
    allButGivenMovieList = list(set(range(MoviesinLatentSpace_Matrix.shape[0]))-set([givenMovieidIndex]))
    distancesTest = relevanceFeedback.euclideanMatrixVector(MoviesinLatentSpace_Matrix[allButGivenMovieList],moviePoint)
    nearestMoviesDistanceTest = np.sort(distancesTest[0])[:r]
    nearestMovieIndicesTest = np.argsort(distancesTest[0])
    nearestMoviesTest = [moviesListTest[i] for i in np.array(nearestMovieIndicesTest)[0]][:r]
    
    return nearestMovies,nearestMoviesTest,nearestMoviesDistance,nearestMoviesDistanceTest
    
