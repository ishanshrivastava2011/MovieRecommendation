from computations import relevanceFeedback as rf
from data import DataHandler
import time


def task1_2LDA(userid):
    movieid_name_map = DataHandler.movieid_name_map
    enter_userid = userid  # input("UserID : ")
    userId = int(enter_userid)
    times = time.time()
    DataHandler.vectors()
    DataHandler.createDictionaries1()
    rf.loadBase(userId)
    movies = rf.runLDADecomposition()
    named_movies = [movieid_name_map[i] for i in movies]
    print('Top 5 movies : ' + str(named_movies))


    while True:
        feedback = input("Relevance (1/0) for each of the 5 movies: ")
        if feedback == 'exit':
            print("GoodBye........")
            break
        feedback = [int(i) for i in feedback.split(',')]
        new_query = rf.newQueryFromFeedBackLDA(movies, feedback)
        print([movieid_name_map[rf.nonwatchedList[i]] for i in new_query][0:5])
        # print(str(new_query) + "\n")

def task1_2CombinedPredictor(userid):
    movieid_name_map = DataHandler.movieid_name_map
    enter_userid = userid  # input("UserID : ")
    userId = int(enter_userid)
    times = time.time()
    DataHandler.vectors()
    DataHandler.createDictionaries1()
    rf.loadBase(userId)
    similarities = rf.runAllMethods()
    movies = [rf.nonwatchedList[i] for i in similarities][0:5]

    named_movies = [movieid_name_map[i] for i in movies]
    print('Top 5 movies : ' + str(named_movies))
    while True:
        feedback = input("Relevance (1/0) for each of the 5 movies: ")
        if feedback == 'exit':
            print("GoodBye........")
            break
        feedback = [int(i) for i in feedback.split(',')]
        new_query = rf.runAllMethodrelevancefeedback(movies, feedback)
        print([movieid_name_map[rf.nonwatchedList[i]] for i in new_query][0:5])
        # print(str(new_query) + "\n")


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
    movies = rf.recommendMovies(new_query)
    named_movies = [movieid_name_map[i] for i in movies]
    print('Top 5 movies : ' + str(named_movies))
    while True:
        feedback = input("Relevance (1/0) for each of the 5 movies: ")
        if feedback == 'exit':
            print("GoodBye........")
            break
        feedback = [int(i) for i in feedback.split(',')]
        new_query = rf.newQueryFromFeedBack(movies, feedback)
        print([movieid_name_map[rf.nonwatchedList[i]] for i in new_query][0:5])
        # print(str(new_query) + "\n")

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
    task1_2CombinedPredictor(userid)
