from computations import relevanceFeedback as rf
from data import DataHandler
import time
from computations import tasksBusiness as tb
import numpy as np
from operator import itemgetter

def task1_2CombinedPredictor(userid):
    movieid_name_map = DataHandler.movieid_name_map
    enter_userid = userid  # input("UserID : ")
    userId = int(enter_userid)
    times = time.time()
    DataHandler.vectors()
    DataHandler.createDictionaries1()
    rf.loadBase(userId)
    similarities, sortedSimilarity = rf.runAllMethods(userid)
    movies = [rf.nonwatchedList[i] for i in similarities][0:5]
    moviesWatched_timestamp = list(DataHandler.user_rated_or_tagged_date_map.get(userId))
    
    moviesWatched_timestamp = sorted(moviesWatched_timestamp,key=itemgetter(1))
    moviesWatched_timestamp_sorted = list(list(zip(*moviesWatched_timestamp ))[0])
    watchedMovieNames = [movieid_name_map[movieid] for movieid in moviesWatched_timestamp_sorted]
    print('-------------------------------------')
    print('Movies Watched by the user in order: '+ str(watchedMovieNames))
    named_movies = [movieid_name_map[i] for i in movies]
    print('Top 5 movies : ' + str(list(zip(named_movies, sortedSimilarity))))
    print('-------------------------------------')
    while True:
        feedback = input("Relevance (1/0) for each of the 5 movies: ")
        if feedback == 'exit':
            print("Exit........")
            break
        feedback = [int(i) for i in feedback.split(',')]
        new_query = rf.runAllMethodrelevancefeedback(movies, feedback)
        print([movieid_name_map[rf.nonwatchedList[i]] for i in new_query][0:5])


def task1_2Decompostions(func, userid):
    movieid_name_map = DataHandler.movieid_name_map
    enter_userid = userid  # input("UserID : ")
    userId = int(enter_userid)
    times = time.time()
    DataHandler.vectors()
    DataHandler.createDictionaries1()
    rf.loadBase(userId)
    rf.runDecomposition(func)

    new_query = rf.q_vector
    movies, distances = rf.recommendMovies(new_query)
    named_movies = [movieid_name_map[i] for i in movies]
    print('---------------------')
    print('Top 5 movies : ')
    print (str(list(zip (named_movies,distances))))
    #for i in range(0, len(named_movies)):
     #   print(named_movies[i] + ", " + str(distances[i]))
    print("---------------------")
    while True:
        feedback = input("Relevance (1/0) for each of the 5 movies: ")
        if feedback == 'exit':
            print("Exit........")
            break
        feedback = [int(i) for i in feedback.split(',')]
        new_query, weights = rf.newQueryFromFeedBack(movies, feedback)
        # print(str(new_query) + "\n")
        print([movieid_name_map[rf.nonwatchedList[i]] for i in new_query][0:5])

def task1_2PCA():
    userid = input("UserID : ")
    task1_2Decompostions(rf.loadPCASemantics, userid)


def task1_2SVD():
    userid = input("UserID : ")
    task1_2Decompostions(rf.loadSVDSemantics, userid)


def task1_2CP():
    userid = input("UserID : ")
    task1_2Decompostions(rf.loadCPSemantics, userid)


def task1_2Combined():
    userid = input("UserID : ")
    task1_2CombinedPredictor(int(userid))
    
def task1_2PageRank():
    userid = input("UserID : ")
    DataHandler.vectors()
    enter_userid = userid  # input("UserID : ")
    userId = int(enter_userid)
    DataHandler.createDictionaries1()
    rf.loadBase(userId);
    rf.task1d(userId)
    
def task3() :
    tb.task3();

def task1_2LDA():
    userid = input("UserID : ")
    movieid_name_map = DataHandler.movieid_name_map
    enter_userid = userid  # input("UserID : ")
    userid = int(enter_userid)
    DataHandler.vectors()
    DataHandler.createDictionaries1()
    rf.loadBase(userid)
    finalWeights = rf.finalWeights
    
    
    movie_movie_similarity_subset_new = rf.runLDADecomposition(userid)#update
    sim = list(movie_movie_similarity_subset_new.T.dot(finalWeights).astype(np.float32))
    movieList = list(movie_movie_similarity_subset_new.columns)
    simSorted = list(np.sort(sim)[::-1])[:5]
    simArgSorted = list(np.argsort(sim)[::-1])
    movies = [movieList[i] for i in simArgSorted][:5]
    named_movies = [movieid_name_map[movie] for movie in movies]
    watchedMovieNames = [movieid_name_map[movieid] for movieid in rf.moviesWatched]
    print(watchedMovieNames)
    print("---------------------------------------------")
    print('Top 5 movies and their similarity scores: \n' +str(list(zip(named_movies,simSorted)))+"\n")
    wantFeedback = True
    while wantFeedback:
        feedbackWant = input("Would you like to give feedback 'Y'/'N': ")
        if feedbackWant == 'Y':
            LDAFeedback(movies)
            wantFeedback = True
        elif feedbackWant == 'N':
            wantFeedback = False
            break
        else:
            print("Invalid Input provided. Please try again.")
            wantFeedback = True
    
def LDAFeedback(movies): 
    takeFeedback = True
    r = len(movies)
    
    feedback = input("Relevance (1/0) for each of the "+ str(r) +" movies: ")
    feedback_split = [int(i) for i in feedback.split(',')]
        
    movieid_name_map = DataHandler.movieid_name_map
    allMovies = sorted(list(movieid_name_map.keys()))
    movieWatchedindex = [allMovies.index(mid) for mid in movies]
    lda_sem_matx=np.matrix(DataHandler.load_movie_LDASpace_df())
    lda_sem_matx_subset = lda_sem_matx[list(set(range(len(allMovies)))-set(movieWatchedindex))]
    
    #new_query = rf.newQueryFromFeedBackLDA(movies, feedback,lda_sem_matx_subset)#update
    new_query_remove = rf.newQueryFromFeedBackLDA(movies, feedback_split,lda_sem_matx_subset)#update
   
    print([movieid_name_map[rf.nonwatchedList[i]] for i in list(np.array(new_query_remove)[0])][0:5])
    
def load_dataForClassifiers():
    return rf.loadPCASemantics()
