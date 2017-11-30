from data import DataHandler
from computations import decompositions
import time
import itertools
from util import constants,formatter
from operator import itemgetter
import numpy as np
from numba import guvectorize, float32,jit
import sklearn.metrics.pairwise as pairwise
import pandas as pd
from computations import personalizedpagerank as ppr
from computations import pickle
from computations import tasksBusiness

movie_movie_similarity = None
moviesWatched_timestamp_sorted = None
moviesWatched = None
q_vector = None
aug_semantic_matx = None
lda_sem_matx = None
moviesList = None
finalWeights = None
nonwatchedList = None
indx = None
q_vectorList = []
sem_matrix_list = []

def listIndex(full_list, sub_list):
    subset = set(sub_list)
    indexList = []
    unwList = []
    for i in range(len(full_list)):
        if full_list[i] in subset:
            indexList.append(i)
        else:
            unwList.append(full_list[i])

    return indexList,unwList

@guvectorize([(float32[:, :], float32[:], float32[:])], '(m,n),(n)->(m)')
def euclideanMatrixVector(matx, vec, distances):
    for i in range(matx.shape[0]):
        distances[i] = np.linalg.norm(matx[i] - vec)

@guvectorize([(float32[:, :], float32[:, :])], '(m,n)->(m,m)')
def euclideanSimilarityMatrix(matx, distances):
    for i in range(matx.shape[0]):
        for j in range(matx.shape[0]):
            distances[i][j] = np.linalg.norm(matx[i] - matx[j])

def loadCPSemantics():
    decomposed = pickle.create_CP_Tensor_pickle()#decompositions.CPDecomposition(DataHandler.getTensor_ActorMovieGenre(), 5)
    fullList = sorted(list(DataHandler.movie_actor_map.keys()))
    tagged_movies_idx = []
    setofmovies = set(moviesList)
    for i in range(len(fullList)):
        if fullList[i] in setofmovies:
            tagged_movies_idx.append(i)

    temp = np.array(decomposed[1])
    return np.take(temp, tagged_movies_idx, axis=0)

def loadPCASemantics():
    movie_tag_df = DataHandler.load_movie_tag_df()
    return pickle.create_PCA_pickle(movie_tag_df)#decompositions.PCADimensionReduction((movie_tag_df), 15)

def loadSVDSemantics():
    movie_tag_df = DataHandler.load_movie_tag_df()
    return pickle.create_SVD_pickle(movie_tag_df) #decompositions.SVDDecomposition((movie_tag_df), 800)[0]

def loadLDASemantics():
    #Load Pickle
    #Return Pickled Semantics
    return None
   # return pd.read_pickle(constants.DIRECTORY + "SVD_decomposition")#decompositions.SVDDecomposition((movie_tag_df), 5)[0]

#def loadLDASemantics():
 #   movie_tag_df = pd.read_pickle(constants.DIRECTORY +"movie_tag_df")#DataHandler.load_movie_tag_df()
  #  return np.array(decompositions.LDADecomposition(movie_tag_df, 5, constants.genreTagsSpacePasses)[1].dense)

def loadBase(userId):
    global moviesWatched
    global finalWeights
    global moviesList
    global nonwatchedList
    global indx

    timestamps = np.array(
        [item[1] for item in list(itertools.chain(*DataHandler.user_rated_or_tagged_date_map.values()))])
    moviesList = sorted(list(DataHandler.movie_tag_map.keys()))
    moviesRated = dict(DataHandler.user_movie_ratings_map.get(userId))
    moviesWatched = sorted(list(DataHandler.user_rated_or_tagged_map.get(userId)))
    moviesWatched_timestamp = sorted(list(DataHandler.user_rated_or_tagged_date_map.get(userId)), key=itemgetter(0))
    moviesWatched_array = np.array([movie[1] for movie in moviesWatched_timestamp])
    indx, nonwatchedList = listIndex(moviesList, moviesWatched)
    time_max = timestamps.max()
    time_min = timestamps.min()
    timestamp_weights = (moviesWatched_array - time_min + 0.00001) \
                        / (time_max - time_min + 0.00001)
    ratingWeights = np.array([moviesRated[movie[0]] for movie in moviesWatched_timestamp])
    finalWeights = timestamp_weights * 0.1 + ratingWeights * 0.9


def runAllMethods(userid):
    global sem_matrix_list
    global q_vectorList

    functions = [loadPCASemantics, loadSVDSemantics, loadCPSemantics]
    allSimilarities = []
    for i in range(1,6):
        similarity=list()
        if(i<=3):
            similarity_semantic_matrix = functions[i-1]()
            similarity_semantic_matrix = ((similarity_semantic_matrix - similarity_semantic_matrix.min(axis=0) + 0.00001) \
                            / (similarity_semantic_matrix.max(axis=0) - similarity_semantic_matrix.min(axis=0) + 0.00001))
    
            vector = np.take(similarity_semantic_matrix, indx, axis=0)
            q_vector = vector.astype(np.float32)
    
            aug_sim_matx = np.delete(similarity_semantic_matrix, indx, axis=0).astype(np.float32)
            sem_matrix_list.append(aug_sim_matx)
            q_vectorList.append(q_vector)
    
            distance = []
            for v in q_vector:
                distance.append(euclideanMatrixVector(aug_sim_matx, v))
    
            distance = np.array(distance)
            distance = (distance - distance.min() + 0.00001) / ((distance.max() - distance.min() + 0.00001))
            similarity = 1. / distance + 0.00001
            similarity = list(similarity.T.dot(finalWeights).astype(np.float32))
        if i==4:
            movie_movie_similarity_subset_new = runLDADecomposition(userid)#update
            similarity = list(movie_movie_similarity_subset_new.T.dot(finalWeights).astype(np.float32))
        if i == 5:
            movieRatedSeed = list(zip(moviesWatched, finalWeights))#DataHandler.userMovieOrders(userId)
            P = DataHandler.load_movie_tag_df()#DataHandler.load_movie_tag_df()
            moviesList = sorted(list(DataHandler.movie_actor_rank_map.keys()))
            euclidean_distance = pairwise.euclidean_distances(P)
            epsilon = np.matrix(np.zeros(euclidean_distance.shape) + 0.000001)
            movie_movie_similarity = 1/(epsilon + euclidean_distance)
            movie_movie_similarity = pd.DataFrame(movie_movie_similarity)
            prData = ppr.personalizedPageRankWeighted(movie_movie_similarity, movieRatedSeed, 0.9)
            moviesNotWATCHED = list(set(moviesList)-set(moviesWatched))
            moviesNotWATCHED_indices = [moviesList.index(i) for i in moviesNotWATCHED]
            similarity = list(prData.loc[moviesNotWATCHED_indices,][0])
        similarity=[formatter.normalizer(min(similarity), max(similarity), value) for value in similarity]
        
        allSimilarities.append(similarity)

    similarities = np.array(allSimilarities).mean(axis=0)

    return np.argsort(similarities)[::-1], np.sort(similarities)[::-1]

def runAllMethodrelevancefeedback(recommended_movies, feedback):
    global nonwatchedList
    global sem_matrix_list

    votes = []

    for aug_sim_matx in sem_matrix_list:
        recommended_idx = [aug_sim_matx[nonwatchedList.index(i)] for i in recommended_movies]

        relevant = np.sum([recommended_idx[i] for i in range(len(recommended_idx)) if feedback[i] == 1], axis=0)
        non_relevant = np.sum([recommended_idx[i] for i in range(len(recommended_idx)) if feedback[i] == 0], axis=0)
        # n_corpus = aug_sim_matx.sum(axis=0)
        n_N = non_relevant/5
        p_vector = (relevant)*(1.0/(len(relevant) + 1.0))
        u_vector = (non_relevant)*(1.0 /(len(non_relevant)+1.0))

        new_q = ((p_vector*(1 - u_vector))/(u_vector*(1 - p_vector)))
        vals = np.power(new_q, aug_sim_matx)
        product = np.prod(vals, axis=1)
        product = (product - product.min() + 0.00001) / ((product.max() - product.min() + 0.00001))

        votes.append(product)

    return np.argsort(np.mean(votes, axis=0))[::-1]


def runLDADecomposition(userid):
    return tasksBusiness.LDA_SIM(userid)


def runDecomposition(func):
    global q_vector
    global aug_semantic_matx
    global similarity_semantic_matrix

    similarity_semantic_matrix = func()
    similarity_semantic_matrix = ((similarity_semantic_matrix - similarity_semantic_matrix.min(axis=0) + 0.00001) \
     / (similarity_semantic_matrix.max(axis=0) - similarity_semantic_matrix.min(axis=0) + 0.00001))


    vector = np.take(similarity_semantic_matrix, indx, axis=0)
    # q_vector = vector.T.dot(finalWeights).astype(np.float32)
    q_vector = vector.astype(np.float32)

    aug_semantic_matx = np.take(similarity_semantic_matrix, indx, axis=0).astype(np.float32)

    return aug_semantic_matx


def execute_query(q_vector):
    global aug_semantic_matx
    times = time.time()
    distance = []
    for vector in q_vector:
        distance.append(euclideanMatrixVector(aug_semantic_matx, vector))
    distance = np.array(distance) + 0.000001    
    distance = 1./distance
    distance = distance.T.dot(finalWeights).astype(np.float32)

    print(' query ---- ' + str(time.time() - times) + ' ---- query')
    return np.argsort(distance)[::-1], np.sort(distance)[::-1]


def recommendMovies(q_vector):
    global nonwatchedList
    distances, actualDistances = execute_query(q_vector)

    movieid_name_map = DataHandler.movieid_name_map
    watchedMovieNames = [movieid_name_map[movieid] for movieid in moviesWatched]
    print(watchedMovieNames)
    return [nonwatchedList[i] for i in distances][0:5], actualDistances[0:5]


def newQueryFromFeedBackLDA(recommended_movies, feedback,lda_sem_matx):
    global nonwatchedList
    

    recommended_idx = [(lda_sem_matx[nonwatchedList.index(i)]) for i in recommended_movies]


    relevant = np.sum([recommended_idx[i] for i in range(len(recommended_idx)) if int(feedback[i]) == 1], axis=0)
    non_relevant = np.sum([recommended_idx[i] for i in range(len(recommended_idx)) if int(feedback[i]) == 0], axis=0)
    # n_corpus = aug_sim_matx.sum(axis=0)
    n_N = non_relevant/5
    p_vector = (relevant)*(1.0/(len(relevant) + 1.0))
    u_vector = (non_relevant)*(1.0 /(len(non_relevant)+1.0))

    new_q = ((p_vector*(1 - u_vector))/1+(u_vector*(1 - p_vector)))
    vals = np.power(new_q, lda_sem_matx)
    #vals[vals <0.0011 ] = 0.5
    #return np.argsort(np.max(vals, axis=1))[::-1]
    return np.argsort(vals[0])[::-1]


def newQueryFromFeedBack(recommended_movies, feedback):
    global nonwatchedList
    global aug_semantic_matx

    recommended_idx = [aug_semantic_matx[nonwatchedList.index(i)] for i in recommended_movies]

    relevant = np.sum([recommended_idx[i] for i in range(len(recommended_idx)) if feedback[i] == 1], axis=0)
    non_relevant = np.sum([recommended_idx[i] for i in range(len(recommended_idx)) if feedback[i] == 0], axis=0)
    # n_corpus = aug_sim_matx.sum(axis=0)
    
    R = 0
    NR = 0
    try:
        R=len(relevant)
    except:
        R=0+0.000001
    
    try:
        NR = len(non_relevant)
    except:
        NR = 0+0.00001
    
    n_N = non_relevant/5
    p_vector = (relevant+0.000001)*(1.0/(R + 1.0))
    u_vector = (non_relevant+0.000001)*(1.0 /(NR+1.0))

    new_q = ((p_vector*(1 - u_vector))/(u_vector*(1 - p_vector)))
    vals = np.power(new_q, aug_semantic_matx)
    products = np.prod(vals, axis=1)

    return np.argsort(products)[::-1], np.sort(products)[::-1]


def runme():
    global q_vector
    movieid_name_map = DataHandler.movieid_name_map
    enter_userid = 36  # input("UserID : ")
    userId = int(enter_userid)
    times = time.time()
    DataHandler.vectors()
    DataHandler.createDictionaries1()
    loadBase(userId)
    # runDecomposition(loadPCASemantics)

    distances = runAllMethods()
    reco = [nonwatchedList[i] for i in distances][0:5]
    runAllMethodrelevancefeedback(reco, [1,1,1,0,0])
    new_query = q_vector
    movies = recommendMovies(new_query)
    named_movies = [movieid_name_map[i] for i in movies]
    print('Top 5 movies : ' + str(named_movies))
    while True:
        feedback = input("Relevance (1/0) for each of the 5 movies: ")
        if feedback == 'exit':
            print("GoodBye........")
            break
        feedback = [int(i) for i in feedback.split(',')]
        new_query = newQueryFromFeedBack(movies,feedback)
        print([movieid_name_map[nonwatchedList[i]] for i in new_query][0:5])
        # print(str(new_query) + "\n")

def newQueryFromRochioFeedBack(moviepoint, relevantMovieList, irrelevantMovieList, MoviesinLatentSpace):
    relevantSum, irrevelantSum = returnSumOfRel_SumNonRel(relevantMovieList, irrelevantMovieList, MoviesinLatentSpace)
    if  len(relevantMovieList) == 0 :
         factor = constants.GAMMA*(irrevelantSum/len(irrelevantMovieList))
    elif len(irrelevantMovieList) == 0 :
         factor = constants.BETA*(relevantSum/len(relevantMovieList))
    else :
         factor = constants.BETA*(relevantSum/len(relevantMovieList)) - constants.GAMMA*(irrevelantSum/len(irrelevantMovieList))
    return (moviepoint + factor).astype(np.float32)

def returnSumOfRel_SumNonRel(Rel,NonRel,MoviesinLatentSpace):
    MoviesinLatentSpace_Matrix = np.matrix(MoviesinLatentSpace,dtype = np.float32)
    moviesList =list( MoviesinLatentSpace.index)
    if len(Rel) == 0 :
        sumRelPoint = [0]*len(MoviesinLatentSpace.columns)
    else:
        Rel_movieIndices = [moviesList.index(mid) for mid in Rel]
        sumRelPoint = MoviesinLatentSpace_Matrix[Rel_movieIndices].sum(axis = 0)
    if len(NonRel) == 0:
        sumNonRelPoint = [0]*len(MoviesinLatentSpace.columns)
    else :
        NonRel_movieIndices = [moviesList.index(mid) for mid in NonRel]
        sumNonRelPoint = MoviesinLatentSpace_Matrix[NonRel_movieIndices].sum(axis = 0)
    return sumRelPoint,sumNonRelPoint
    
def newQueryFromLDEDecHiFeedBack(moviePoint, relevantMovieList, irrevelantMovieList, nearestMovies, MoviesinLatentSpace):
    if len(irrevelantMovieList) != 0:
        sumRelPoint,sumNonRelPoint = returnSumOfRel_SumNonRel(relevantMovieList,[irrevelantMovieList[-1]],MoviesinLatentSpace)
    else:
        sumRelPoint,sumNonRelPoint = returnSumOfRel_SumNonRel(relevantMovieList,irrevelantMovieList,MoviesinLatentSpace)
    return (moviePoint + sumRelPoint - sumNonRelPoint).astype(np.float32)

def newQueryFromLDERegularFeedBack(moviePoint, relevantMovieList, irrevelantMovieList, nearestMovies, MoviesinLatentSpace):
    sumRelPoint,sumNonRelPoint = returnSumOfRel_SumNonRel(relevantMovieList,irrevelantMovieList,MoviesinLatentSpace)
    return (moviePoint + sumRelPoint - sumNonRelPoint).astype(np.float32)

def task1d(userId) :
    movieRatedSeed = list(zip(moviesWatched, finalWeights))#DataHandler.userMovieOrders(userId)
    P = DataHandler.load_movie_tag_df()#DataHandler.load_movie_tag_df()
    moviesList = sorted(list(DataHandler.movie_actor_rank_map.keys()))
    euclidean_distance = pairwise.euclidean_distances(P)
    epsilon = np.matrix(np.zeros(euclidean_distance.shape) + 0.000001)
    movie_movie_similarity = 1/(epsilon + euclidean_distance)
    movie_movie_similarity = pd.DataFrame(movie_movie_similarity)
    prData = ppr.personalizedPageRankWeighted(movie_movie_similarity, movieRatedSeed, 0.9)
    rankedItems = sorted(list(map(lambda x:(moviesList[x[0]],x[1]),prData.itertuples())),key=lambda x:x[1], reverse=True)
    movieid_name_map = DataHandler.movieid_name_map

    seedmovieNames = [movieid_name_map[k] for k,y in movieRatedSeed]
    print("Movies similar to the users seed movies " + str(seedmovieNames) + " are:")
    required =  sorted([(movieid_name_map[k],y) for (k,y) in rankedItems if k not in [k for k,y in movieRatedSeed]],key=lambda x: x[1], reverse=True)
    print("--------------------------------------")
    print (required[:5])
    
    while True:
        feedback = input("Relevance (1/0) for each of the 5 movies: ")
        if feedback == 'exit':
            print("GoodBye........")
            break
        feedback_split = feedback.split(',')
        feedback = np.array([int(i) for i in feedback_split])
        
        relevantFeedback = np.where(feedback == 1)[0]
        a = required[:5]
        b = [i[0] for i in a]
        
        final =list()
        
        movie_id_map = DataHandler.movie_id_map
        relevantMovieList = [b[i] for i in relevantFeedback]
        movieids = [movie_id_map[k] for k in relevantMovieList]
        final = [1/len(relevantMovieList)] * len(relevantMovieList)
        movieRelevantSeed = list(zip(movieids, final))
        prData1 = ppr.personalizedPageRankWeighted(movie_movie_similarity, movieRelevantSeed, 0.9)
        rankeds = sorted(list(map(lambda x:(moviesList[x[0]],x[1]),prData1.itertuples())),key=lambda x:x[1], reverse=True)
        movieid_name_map = DataHandler.movieid_name_map
        required1 =  sorted([(movieid_name_map[k],y) for (k,y) in rankeds],key=lambda x: x[1], reverse=True)
        print("--------------------------------------")
        print (required1[:5])
        #new_query = newQueryFromFeedBack(movies, feedback)
        #print([movieid_name_map[rf.nonwatchedList[i]] for i in new_query][0:5])
    