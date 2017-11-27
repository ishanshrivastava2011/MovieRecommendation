from data import DataHandler
from computations import decompositions
import time
from util import constants
from operator import itemgetter
import numpy as np
from numba import guvectorize, float32,jit

movie_movie_similarity = None
moviesWatched_timestamp_sorted = None
moviesWatched = None
q_vector = None
aug_sim_matx = None
moviesList = None
nonwatchedList = None

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
    decomposed = decompositions.CPDecomposition(DataHandler.getTensor_ActorMovieGenre(), 5)
    return np.array(decomposed[1])

def loadPCASemantics():
    movie_tag_df = DataHandler.load_movie_tag_df()
    return decompositions.PCADimensionReduction((movie_tag_df.transpose()), 5)

def loadSVDSemantics():
    movie_tag_df = DataHandler.load_movie_tag_df()
    return decompositions.SVDDecomposition((movie_tag_df.transpose()), 5)[0]

def loadLDASemantics():
    movie_tag_df = DataHandler.load_movie_tag_df()
    return np.array(decompositions.LDADecomposition(movie_tag_df.transpose(), 5, constants.genreTagsSpacePasses)[1].dense)


def loadBase(userId, func):
    global movie_movie_similarity
    global moviesWatched_timestamp_sorted
    global moviesWatched
    global q_vector
    global aug_sim_matx
    global moviesList
    global nonwatchedList
    global similarity_semantic_matrix

    DataHandler.vectors()
    DataHandler.createDictionaries1()
    similarity_semantic_matrix = func()
    similarity_semantic_matrix = ((similarity_semantic_matrix - similarity_semantic_matrix.min(axis=0) + 0.00001) \
     / (similarity_semantic_matrix.max(axis=0) - similarity_semantic_matrix.min(axis=0) + 0.00001))


    moviesList = sorted(list(DataHandler.movie_actor_rank_map.keys()))
    moviesWatched = list(DataHandler.user_rated_or_tagged_map.get(userId))
    moviesWatched_timestamp = list(DataHandler.user_rated_or_tagged_date_map.get(userId))
    moviesWatched_timestamp = sorted(moviesWatched_timestamp, key=itemgetter(1))
    moviesWatched_timestamp_sorted = list(list(zip(*moviesWatched_timestamp))[0])
    windx = (np.argsort(moviesWatched_timestamp_sorted)+1)
    windx = windx*(1/np.sum(windx))

    indx,nonwatchedList = listIndex(moviesList, moviesWatched)
    vector = np.take(similarity_semantic_matrix, indx, axis=0)
    q_vector = vector.T.dot(windx).astype(np.float32)
    aug_sim_matx = np.delete(similarity_semantic_matrix, indx, axis=0).astype(np.float32)



def execute_query(q_vector):
    global aug_sim_matx
    global moviesWatched_timestamp_sorted
    times = time.time()
    distances = euclideanMatrixVector(aug_sim_matx, q_vector)
    # print(' query ---- ' + str(time.time() - times) + ' ---- query')
    return np.argsort(distances)


def recommendMovies(q_vector):
    global nonwatchedList
    global moviesWatched_timestamp_sorted
    distances = execute_query(q_vector)

    movieid_name_map = DataHandler.movieid_name_map
    watchedMovieNames = [movieid_name_map[movieid] for movieid in moviesWatched_timestamp_sorted]
    print(watchedMovieNames)
    return [nonwatchedList[i] for i in distances][0:5]


def newQueryFromFeedBack(recommended_movies, feedback):
    global nonwatchedList
    global aug_sim_matx

    recommended_idx = [aug_sim_matx[nonwatchedList.index(i)] for i in recommended_movies]

    relevant = np.sum([recommended_idx[i] for i in range(len(recommended_idx)) if feedback[i] == 1], axis=0)
    n_corpus = aug_sim_matx.sum(axis=0)
    n_N = n_corpus/len(n_corpus)
    p_vector = (relevant + n_N)*(1.0/(len(relevant) + 1.0))
    u_vector = (n_corpus - relevant + n_N)*(1.0 /(len(n_corpus)+1.0))

    n_query = np.log((p_vector*(1 - u_vector))/(u_vector*(1 - p_vector)))
    return n_query


def runme():
    global q_vector
    movieid_name_map = DataHandler.movieid_name_map
    enter_userid = 36  # input("UserID : ")
    userId = int(enter_userid)
    times = time.time()
    loadBase(userId,loadPCASemantics)
    new_query = q_vector
    while True:

        movies = recommendMovies(new_query)
        named_movies = [movieid_name_map[i] for i in movies]
        print('Top 5 movies : ' + str(named_movies))
        feedback = input("Relevance (1/0) for each of the 5 movies: ")
        if feedback == 'exit':
            print("GoodBye........")
            break
        feedback = [int(i) for i in feedback.split(',')]
        new_query = newQueryFromFeedBack(movies,feedback)
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