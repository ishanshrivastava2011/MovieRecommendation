from collections import defaultdict
import pandas as pd
import dateutil
import sys
import time
import math
import csv
from computations import metrics
from computations import decompositions
from util import constants
from util import formatter

import numpy as np
from operator import itemgetter

max_rank = 0
min_rank = sys.maxsize

max_date = 0
min_date = sys.maxsize
tag_count = 0

tagset_genre = defaultdict(set)
actor_weight_vector_tf = dict()
actor_weight_vector_tf_idf = dict()
genre_weight_vector_tf = dict()
genre_weight_vector_tf_idf = dict()
user_tag_map_tf = defaultdict()
user_tag_map_tf_idf = defaultdict()

movie_actor_df = pd.read_csv(constants.DIRECTORY + "movie-actor.csv")
tag_movie_df = pd.read_csv(constants.DIRECTORY + "mltags.csv")
genre_movie_df = pd.read_csv(constants.DIRECTORY + "mlmovies.csv")
tag_id_df = pd.read_csv(constants.DIRECTORY + "genome-tags.csv")
user_ratings_df = pd.read_csv(constants.DIRECTORY + "mlratings.csv")
actor_info_df = pd.read_csv(constants.DIRECTORY + "imdb-actor-info.csv")

actor_movie_map = defaultdict(set)
movie_actor_map = defaultdict(set)
movie_year_map = defaultdict(int)
year_movie_map = defaultdict(int)
movie_ratings_map = defaultdict(list)
uniqueRatings = set()
uniqueRanks = set()
actor_movie_rank_map = defaultdict(set)
movie_actor_rank_map = defaultdict(set)
movie_tag_map = defaultdict(set)
genre_movie_map = defaultdict(set)
user_tag_map = defaultdict(set)
tag_user_map = defaultdict(set)
genre_tagset = defaultdict(set)
tag_movie_map = defaultdict(list)
user_rated_or_tagged_map = defaultdict(set)
tag_id_map = dict()
id_tag_map = dict()
actor_actorid_map = defaultdict(str)
movie_genre_map = defaultdict(set)
movieid_name_map = defaultdict()
tag_timestamp_map=defaultdict()

user_rated_or_tagged_date_map = defaultdict(set)

def create_actor_actorid_map():
    for row in actor_info_df.itertuples():
        actor_actorid_map[row.id]=row.name

def vectors():
    global max_rank
    global min_rank
    global tag_count
    global max_date
    global min_date
    t = time.time()
    
    for row in tag_id_df.itertuples():
        tag_id_map[row.tagId] = row.tag
        id_tag_map[row.tag] = row.tagId
    for row in user_ratings_df.itertuples():
        user_rated_or_tagged_map[row.userid].add(row.movieid)
        user_rated_or_tagged_date_map[row.userid].add(tuple((row.movieid,dateutil.parser.parse(row.timestamp).timestamp())))
            
    tagset = set()
    for row in tag_movie_df.itertuples():
        date_time = dateutil.parser.parse(row.timestamp).timestamp()
        if date_time > max_date:
            max_date = date_time
        if date_time < min_date:
            min_date = date_time
        tagset.add(row.tagid)
        user_rated_or_tagged_map[row.userid].add(row.movieid)
        tag_movie_map[row.tagid].append((row.movieid, date_time))
        user_tag_map[row.userid].add((row.tagid, date_time))
        tag_user_map[row.tagid].add((row.userid, date_time))
        movie_tag_map[row.movieid].add((row.tagid, date_time))
        user_rated_or_tagged_date_map[row.userid].add(tuple((row.movieid,dateutil.parser.parse(row.timestamp).timestamp())))
        tag_timestamp_map[row.tagid]=dateutil.parser.parse(row.timestamp).timestamp()
    tag_count = tagset.__len__()
    tagset.clear()
    print('Main : ', time.time() - t)

def getGenreMoviesMap():
    genre_movies_map = {}
    for row in genre_movie_df.itertuples():
        genres_list = row.genres.split("|")
        for genre in genres_list:
            if (genre in genre_movies_map.keys()):
                genre_movies_map[genre].append(row.movieid)
            else:
                genre_movies_map[genre] = [row.movieid]
    return genre_movies_map

def createDictionaries1():
    global max_rank
    global min_rank
    global tag_count
    global max_date
    global min_date
    for row in movie_actor_df.itertuples():
        if row.actor_movie_rank < min_rank:
            min_rank = row.actor_movie_rank
        if row.actor_movie_rank > max_rank:
            max_rank = row.actor_movie_rank
        actor_movie_rank_map[row.actorid].add((row.movieid, row.actor_movie_rank))
        movie_actor_rank_map[row.movieid].add((row.actorid, row.actor_movie_rank))
        actor_movie_map[row.actorid].add((row.movieid))
        movie_actor_map[row.movieid].add((row.actorid))
        uniqueRanks.add(row.actor_movie_rank)
    
    for row in genre_movie_df.itertuples():
        genres_list = row.genres.split("|")
        for genre in genres_list:
            genre_movie_map[genre].add(row.movieid)
            movie_genre_map[row.movieid].add(genre)
        movie_year_map[row.movieid]=row.year
        year_movie_map[row.year]=row.movieid
        movieid_name_map[row.movieid]=row.moviename
    for row in user_ratings_df.itertuples():
        movie_ratings_map[row.movieid].append(row.rating)
        uniqueRatings.add(row.rating)
# def load_genre_count_matrix(given_genre):
# 	for row in genre_movie_df.itertuples():
# 		genres_list = row.genres.split("|")
# 		for genre in genres_list:
# 			genre_movie_map[genre].add(row.movieid)
#
# 	tagList = sorted(list(tag_movie_map.keys()))
# 	for movie in genre_movie_map[given_genre]:
# 		tag_count_movie = dict([(tag, len(filter(lambda x: x == movie, tag_movie_map[tag]))) for tag in tagList])


def load_genre_matrix(given_genre):
	movieCount = movie_tag_map.keys().__len__()
	createDictionaries1()

	tagList = sorted(list(tag_movie_map.keys()))
	movieList = []
	df = pd.DataFrame(columns=tagList)
	for movie in genre_movie_map[given_genre]:
		tagsInMovie = movie_tag_map[movie]
		tf_idf_map = dict()
		if tagsInMovie:
			movieList.append(movie)
			for tag in tagList:
				moviesInTagCount = len(tag_movie_map[tag])
				tf_numerator = 0
				for temp_movie, datetime in tag_movie_map[tag]:
					if movie == temp_movie:
						tf_numerator += formatter.normalizer(min_date, max_date, datetime)
				tf = tf_numerator / len(tagsInMovie)
				tf_idf = tf * math.log2(movieCount / moviesInTagCount)
				tf_idf_map[tag] = tf_idf
			df = df.append(tf_idf_map, ignore_index=True)
	df.index = movieList
	return df

def load_genre_matrix_tf(given_genre):
	createDictionaries1()

	tagList = sorted(list(tag_movie_map.keys()))
	movieList = []
	df = pd.DataFrame(columns=tagList)
	for movie in genre_movie_map[given_genre]:
		tagsInMovie = movie_tag_map[movie]
		tf_idf_map = dict()
		if tagsInMovie:
			movieList.append(movie)
			for tag in tagList:
				moviesInTagCount = len(tag_movie_map[tag])
				tf_numerator = 0
				for temp_movie, datetime in tag_movie_map[tag]:
					if movie == temp_movie:
						tf_numerator += 1
				tf = tf_numerator
				tf_idf = tf
				tf_idf_map[tag] = tf_idf
			df = df.append(tf_idf_map, ignore_index=True)
	df.index = movieList
	return df

def load_genre_actor_matrix(given_genre):
	global max_rank
	global min_rank
	global tag_count
	global max_date
	global min_date

	createDictionaries1()

	actorList = sorted(list(actor_movie_rank_map.keys()))
	df = pd.DataFrame(columns=actorList)
	movieCount = movie_tag_map.keys().__len__()
	movieList = []

	for movieInGenre in genre_movie_map[given_genre]:
		movieList.append(movieInGenre)
		actorsInMovieList = movie_actor_rank_map[movieInGenre]
		actorCountOfMovie = len(actorsInMovieList)
		tf_idf_map = dict.fromkeys(actorList, 0.0)
		for actor, rank in actorsInMovieList:
			movieCountOfActor = len(actor_movie_rank_map[actor])
			tf_numerator = (1 / formatter.normalizer(min_rank, max_rank, rank))
			tf_idf = (tf_numerator / actorCountOfMovie) * math.log2(movieCount / movieCountOfActor)
			tf_idf_map[actor] = tf_idf
		df = df.append(tf_idf_map, ignore_index=True)
	df.index = movieList
	return df


def actor_actor_similarity_matrix():
	actor_tag_map = actor_tagVector()
	dfList = []
	actorList = sorted(list(actor_tag_map.keys()))
	for actor1 in actorList:
		actorMap = dict.fromkeys(actorList, 0.0)
		for actor2 in actorList:
			vec1 = dict(actor_tag_map[actor1])
			vec2 = dict(actor_tag_map[actor2])
			actorMap[actor2] = metrics.euclideanDistance(vec1, vec2)
		# df.at[actor1,actor2] = metrics.euclidean(vec1,vec2)
		dfList.append(actorMap)
	return pd.DataFrame(dfList, columns=actorList, index=actorList), actorList

def actor_actor_invSimilarity_matrix():
    actor_tag_map = actor_tagVector()
    dfList = []
    actorList = sorted(list(actor_tag_map.keys()))
    for actor1 in actorList:
        actorMap = dict.fromkeys(actorList, 0.0)
        for actor2 in actorList:
            vec1 = dict(actor_tag_map[actor1])
            vec2 = dict(actor_tag_map[actor2])
            adjEuclidean =  constants.EPSILON + metrics.euclideanDistance(vec1, vec2)
            actorMap[actor2] = 1.0/adjEuclidean
		# df.at[actor1,actor2] = metrics.euclidean(vec1,vec2)
        dfList.append(actorMap)
    return pd.DataFrame(dfList, columns=actorList, index=actorList)

def coactor_siilarity_matrix():
	createDictionaries1()
	dfList = []
	actorList = sorted(list(actor_movie_rank_map.keys()))
	for actor1 in actorList:
		actorMap = dict.fromkeys(actorList, 0.0)
		for actor2 in actorList:
			co_star_movie_set = set.intersection(set(k for (k,y) in actor_movie_rank_map[actor1]), set(k for (k,y) in actor_movie_rank_map[actor2]))
			actorMap[actor2] = co_star_movie_set.__len__()
		dfList.append(actorMap)
	return pd.DataFrame(dfList, columns=actorList, index=actorList), actorList


def actor_tag_df():
	actor_weight_vector_tf_idf = actor_tagVector()
	tagList = sorted(list(tag_movie_map.keys()))
	actorList = sorted(list(actor_movie_rank_map.keys()))
	df = pd.DataFrame(columns=tagList)
	dictList = []

	for actor in actorList:
		actor_tag_dict = dict.fromkeys(tagList,0.0)
		for tag,weight in actor_weight_vector_tf_idf[actor]:
			actor_tag_dict[tag] = weight
		dictList.append(actor_tag_dict)
	df = df.append(dictList,ignore_index=True)
	df.index = actorList
	return df

def actor_similarity_tagVector(actor_id_given):
	actor_weight_vector_tf_idf = actor_tagVector()
	actor_vector = actor_weight_vector_tf_idf[actor_id_given]
	print(list(map(lambda x:tag_id_map[x[0]], sorted(actor_vector,key=lambda x:x[1]))))

	actorsList = actor_movie_rank_map.keys()
	return sorted([(actor, metrics.euclidean(actor_vector, actor_weight_vector_tf_idf[actor])) for actor in actorsList],
				  key=lambda x: x[0])


def actor_similarity_matrix(actor_id_given):
	actor_weight_vector_tf_idf = actor_tagVector()
	tagList = sorted(list(tag_movie_map.keys()))
	actorList = sorted(list(actor_movie_rank_map.keys()))
	df = pd.DataFrame(columns=tagList)
	dictList = []

	for actor in actorList:
		actor_tag_dict = dict.fromkeys(tagList,0.0)
		for tag,weight in actor_weight_vector_tf_idf[actor]:
			actor_tag_dict[tag] = weight
		dictList.append(actor_tag_dict)
	df = df.append(dictList,ignore_index=True)

	df = pd.DataFrame(decompositions.PCADimensionReduction(df,5),index=actorList)
	actor_vector = df.loc[actor_id_given]
	return sorted([(actor, metrics.euclidean(actor_vector, df.loc[actor])) for actor in actorList],
				  key=lambda x: x[0])


def actor_tagVector():
	global max_rank
	global min_rank

	for row in movie_actor_df.itertuples():
		if row.actor_movie_rank < min_rank:
			min_rank = row.actor_movie_rank
		if row.actor_movie_rank > max_rank:
			max_rank = row.actor_movie_rank
		actor_movie_rank_map[row.actorid].add((row.movieid, row.actor_movie_rank))
		movie_actor_rank_map[row.movieid].add((row.actorid, row.actor_movie_rank))

	total_actor_count = len(actor_movie_rank_map)
	for actorID, movies_list in actor_movie_rank_map.items():

		tag_counter = 0
		tag_weight_tuple_tf = defaultdict(float)
		tag_weight_tuple_tf_idf = defaultdict(float)
		for movie in movies_list:
			tag_counter += len(movie_tag_map[movie[0]])

		for movieID, rank in movies_list:
			if movieID in movie_tag_map:
				for tag_id, timestamp in movie_tag_map[movieID]:
					actor_count = 0
					aSetOfTags = set()
					for mov in tag_movie_map[tag_id]:
						aSetOfTags.update([k for (k, v) in movie_actor_rank_map[mov[0]]])
					actor_count = aSetOfTags.__len__()
					tf = (formatter.normalizer(min_date, max_date, timestamp)
						  / formatter.normalizer(min_rank, max_rank, rank)) / tag_counter
					tag_weight_tuple_tf[tag_id] += tf
					tag_weight_tuple_tf_idf[tag_id] += tf * math.log2(
						total_actor_count / actor_count)
		actor_weight_vector_tf_idf[actorID] = [(k, v) for k, v in tag_weight_tuple_tf_idf.items()]

	return actor_weight_vector_tf_idf
def get_dicts():
    vectors()
    createDictionaries1()
    return movie_tag_map,tag_id_map,actor_movie_rank_map,movie_actor_rank_map
"""
This function returns an Actor-Movie-Year Tensor.
It creates this tensor by iterating through all the actors and then for each actor,
it iterates through the actor's movies and assigns 1 to this actor, movie and movie's year index triple.
"""
def getTensor_ActorMovieYear():
    createDictionaries1()
    actors = sorted(list(actor_movie_map.keys()))
    movies = sorted(list(movie_actor_map.keys()))
    years = sorted(list(year_movie_map.keys()))
    a = len(actors)
    m = len(movies)
    y = len(years)
    tensor_ActorMovieYear = np.zeros(a*m*y).reshape(a,m,y)
    for actor in actors:
        for movie in actor_movie_map.get(actor):
            tensor_ActorMovieYear[actors.index(actor),movies.index(movie),years.index(movie_year_map.get(movie))] = 1
    return tensor_ActorMovieYear, actors, movies, years


"""
This function returns an Tag-Movie-Rating Tensor.
It creates this tensor by iterating through all the movies in MLtags and then for each movie,
it calculates ithe movies's average rating and then iterates through the movie's tags 
and assign 1 to this movie, tag and ratings>=Average rating index triple.
NOTE:
Out of 86 Unique movies in MLRatings, only 27 of them have been tagged
All the movies that are tagged have ratings for them.
So it doesn't make sense to include movies without any tags in our tensor. 
Therefore I am considering only movies from MLtags.csv
"""
def getTensor_TagMovieRating():
    createDictionaries1()
    vectors()
    tags = sorted(list(tag_movie_map.keys()))
    movies = sorted(list(movie_tag_map.keys()))
    ratings = list(uniqueRatings)
    t = len(tags)
    m = len(movies)
    r = len(ratings)
    tensor_TagMovieRating = np.zeros(t*m*r).reshape(t,m,r)
    for movie in movies:
        movieRatings = movie_ratings_map.get(movie)
        movieAvgRating = sum(movieRatings) / float(len(movieRatings))
        for tag,date in movie_tag_map.get(movie):
            tensor_TagMovieRating[tags.index(tag),movies.index(movie),range(math.ceil(movieAvgRating),r)] = 1
    return tensor_TagMovieRating, tags, movies, ratings

def docSpecificCorpus(df,actorIndex):
    import gensim
    numpy_matrix = np.matrix(df.loc[actorIndex].as_matrix())
    numpy_matrix_transpose = numpy_matrix.transpose()
    corpus = gensim.matutils.Dense2Corpus(numpy_matrix_transpose)
    return list(corpus)[0]

def representDocInLDATopics(df,actorIndex,ldaModel):
    actorInLDATopics = ldaModel[docSpecificCorpus(df,actorIndex)]
    totalTopics = 4
    CurTopics = zip(*actorInLDATopics)
    CurTopics = list(CurTopics)
    for i in range(0,totalTopics):
            if(i not in CurTopics[0]):
                actorInLDATopics.append(tuple((i,0)))
    return actorInLDATopics

def similarActors_LDA(givenActor):
    createDictionaries1()
    vectors()
    givenActor_similarity = defaultdict(float)
    actor_weight_vector_tf_idf = actor_tagVector()
    tagList = sorted(list(tag_movie_map.keys()))
    actorList = sorted(list(actor_movie_rank_map.keys()))
    df = pd.DataFrame(columns=tagList)
    dictList = []
    for actor in actorList:
        actor_tag_dict = dict.fromkeys(tagList,0.0)
        for tag,weight in actor_weight_vector_tf_idf[actor]:
            actor_tag_dict[tag] = weight
        dictList.append(actor_tag_dict)
    df = df.append(dictList,ignore_index=True)
    t = time.time()
    ldaModel,doc_term_matrix,id_Term_map  =  decompositions.LDADecomposition(df,4,constants.actorTagsSpacePasses)
    print('Query : ', time.time() - t)
    for otherActor in actorList:
        ac1 = representDocInLDATopics(df,actorList.index(givenActor),ldaModel)
        if otherActor != givenActor:
            ac2 = representDocInLDATopics(df,actorList.index(otherActor),ldaModel)
            givenActor_similarity[otherActor]=(metrics.simlarity_kullback_leibler(ac1,ac2))
    #print(sorted(givenActor_similarity.items(),key = itemgetter(1),reverse=True))
    top10 = sorted(givenActor_similarity.items(),key = itemgetter(1),reverse=False)[0:11]
    return top10


def load_movie_tag_df():
    movieCount = movie_tag_map.keys().__len__()
    createDictionaries1()

    tagList = list(tag_movie_map.keys())
    movieList = []
    df = pd.DataFrame(columns=tagList)
    for movie in movie_tag_map.keys():
        tagsInMovie = movie_tag_map[movie]
        tf_idf_map = dict()
        if tagsInMovie:
            movieList.append(movie)
            for tag in tagList:
                moviesInTagCount = len(tag_movie_map[tag])
                tf_numerator = 0
                for temp_movie, datetime in tag_movie_map[tag]:
                    if movie == temp_movie:
                        tf_numerator += formatter.normalizer(min_date, max_date, datetime)
                tf = tf_numerator / len(tagsInMovie)
                tf_idf = tf * math.log2(movieCount / moviesInTagCount)
                tf_idf_map[tag] = tf_idf
            df = df.append(tf_idf_map, ignore_index=True)
    df.index = movieList
    return df    

def movie_movie_Similarity(movie_tag_df):
    movies = movie_tag_df.index
    dfList = []
    for movie1 in movies:
        movieMap = dict.fromkeys(movies, 0.0)
        for movie2 in movies:
            vec1 = dict(zip(movie_tag_df.loc[movie1].index,movie_tag_df.loc[movie1]))
            vec2 = dict(zip(movie_tag_df.loc[movie2].index,movie_tag_df.loc[movie2]))
            movieMap[movie2] = metrics.euclidean(vec1, vec2)
        dfList.append(movieMap)
    return pd.DataFrame(dfList, columns=movies, index=movies)

def movie_movie_Similarity1(movie_tag_df):
    movies = movie_tag_df.index
    dfList = []
    for movie1 in movies:
        movieMap = dict.fromkeys(movies, 0.0)
        for movie2 in movies:
            vec1 = dict(zip(movie_tag_df.loc[movie1].index,movie_tag_df.loc[movie1]))
            vec2 = dict(zip(movie_tag_df.loc[movie2].index,movie_tag_df.loc[movie2]))
            movieMap[movie2] = 1/(constants.EPSILON+metrics.euclideanDistance(vec1, vec2))
        dfList.append(movieMap)
    return pd.DataFrame(dfList, columns=movies, index=movies)

def getTensor_ActorMovieGenreYear():
    createDictionaries1()
    actors = sorted(list(actor_movie_map.keys()))
    movies = sorted(list(movie_actor_map.keys()))
    genres = sorted(list(genre_movie_map.keys()))
    years = sorted(list(year_movie_map.keys()))
    a = len(actors)
    m = len(movies)
    g = len(genres)
    y = len(years)
    tensor_ActorMovieGenreYear = np.zeros(a*m*g*y).reshape(a,m,g,y)
    for actor in actors:
        for movie,rank in actor_movie_rank_map.get(actor):
            ratings = movie_ratings_map.get(movie)
            avgRating = sum(ratings)/len(ratings)
            genreMovie = movie_genre_map.get(movie)
            weight = avgRating/rank
            for genre in genreMovie:
                tensor_ActorMovieGenreYear[actors.index(actor),movies.index(movie),genres.index(genre),years.index(movie_year_map.get(movie))] = 1#weight
    return tensor_ActorMovieGenreYear

def getTensor_ActorMovieGenre():
    createDictionaries1()
    actors = sorted(list(actor_movie_map.keys()))
    movies = sorted(list(movie_actor_map.keys()))
    genres = sorted(list(genre_movie_map.keys()))
    a = len(actors)
    m = len(movies)
    g = len(genres)
    tensor_ActorMovieGenreYear = np.zeros(a*m*g).reshape(a,m,g)
    for actor in actors:
        for movie,rank in actor_movie_rank_map.get(actor):
            # ratings = movie_ratings_map.get(movie)
            # avgRating = sum(ratings)/len(ratings)
            genreMovie = movie_genre_map.get(movie)
            weight = 1/rank
            for genre in genreMovie:
                tensor_ActorMovieGenreYear[actors.index(actor),movies.index(movie),genres.index(genre)] = weight
    return tensor_ActorMovieGenreYear

def getTensor_ActorMovieGenreYearRankRating():
    createDictionaries1()
    actors = sorted(list(actor_movie_map.keys()))
    movies = sorted(list(movie_actor_map.keys()))
    genres = sorted(list(genre_movie_map.keys()))
    years = sorted(list(year_movie_map.keys()))
    ranks = list(uniqueRanks)
    ratings = list(uniqueRatings)
    a = len(actors)
    m = len(movies)
    g = len(genres)
    y = len(years)
    rk = len(ranks)
    rt = len(ratings)
    tensor_ActorMovieGenreYearRankRating = np.zeros(a*m*g*y*rk*rt).reshape(a,m,g,y,rk,rt)
    for actor in actors:
        for movie,rank in actor_movie_rank_map.get(actor):
            ratings = movie_ratings_map.get(movie)
            avgRating = sum(ratings)/len(ratings)
            genreMovie = movie_genre_map.get(movie)
            for genre in genreMovie:
                tensor_ActorMovieGenreYearRankRating[actors.index(actor),movies.index(movie),genres.index(genre),years.index(movie_year_map.get(movie)),ranks.index(rank),range(math.ceil(avgRating),rt)] = 1
    return tensor_ActorMovieGenreYearRankRating
def similarActors_LDA_tf(givenActor):
    createDictionaries1()
    vectors()
    givenActor_similarity = defaultdict(float)
    actor_weight_vector_tf = actor_tagVector_tf()
    tagList = sorted(list(tag_movie_map.keys()))
    actorList = sorted(list(actor_movie_rank_map.keys()))
    df = pd.DataFrame(columns=tagList)
    dictList = []
    for actor in actorList:
        actor_tag_dict = dict.fromkeys(tagList,0.0)
        for tag,weight in actor_weight_vector_tf[actor]:
            actor_tag_dict[tag] = weight
        dictList.append(actor_tag_dict)
    df = df.append(dictList,ignore_index=True)
    t = time.time()
    ldaModel,doc_term_matrix,id_Term_map  =  decompositions.LDADecomposition(df,4,constants.actorTagsSpacePasses)
    print('Query : ', time.time() - t)
    for otherActor in actorList:
        ac1 = representDocInLDATopics(df,actorList.index(givenActor),ldaModel)
        if otherActor != givenActor:
            ac2 = representDocInLDATopics(df,actorList.index(otherActor),ldaModel)
            givenActor_similarity[otherActor]=(metrics.simlarity_kullback_leibler(ac1,ac2))
    #print(sorted(givenActor_similarity.items(),key = itemgetter(1),reverse=True))
    top10 = sorted(givenActor_similarity.items(),key = itemgetter(1),reverse=False)[0:11]
    return top10

def load_genre_actor_matrix_tf(given_genre):
	global max_rank
	global min_rank
	global tag_count
	global max_date
	global min_date

	createDictionaries1()

	actorList = sorted(list(actor_movie_rank_map.keys()))
	df = pd.DataFrame(columns=actorList)
	movieCount = movie_tag_map.keys().__len__()
	movieList = []

	for movieInGenre in genre_movie_map[given_genre]:
		movieList.append(movieInGenre)
		actorsInMovieList = movie_actor_rank_map[movieInGenre]
		actorCountOfMovie = len(actorsInMovieList)
		tf_idf_map = dict.fromkeys(actorList, 0.0)
		for actor, rank in actorsInMovieList:
			movieCountOfActor = len(actor_movie_rank_map[actor])
			tf_numerator = 1
			tf_idf = tf_numerator
			tf_idf_map[actor] = tf_idf
		df = df.append(tf_idf_map, ignore_index=True)
	df.index = movieList
	return df

def actor_tagVector_tf():
	global max_rank
	global min_rank

	for row in movie_actor_df.itertuples():
		if row.actor_movie_rank < min_rank:
			min_rank = row.actor_movie_rank
		if row.actor_movie_rank > max_rank:
			max_rank = row.actor_movie_rank
		actor_movie_rank_map[row.actorid].add((row.movieid, row.actor_movie_rank))
		movie_actor_rank_map[row.movieid].add((row.actorid, row.actor_movie_rank))

	for actorID, movies_list in actor_movie_rank_map.items():

		tag_counter = 0
		tag_weight_tuple_tf = defaultdict(float)
		for movie in movies_list:
			tag_counter += len(movie_tag_map[movie[0]])

		for movieID, rank in movies_list:
			if movieID in movie_tag_map:
				for tag_id, timestamp in movie_tag_map[movieID]:
					actor_count = 0
					aSetOfTags = set()
					for mov in tag_movie_map[tag_id]:
						aSetOfTags.update([k for (k, v) in movie_actor_rank_map[mov[0]]])
					actor_count = aSetOfTags.__len__()
					tf = 1
					tag_weight_tuple_tf[tag_id] += tf					
		actor_weight_vector_tf[actorID] = [(k, v) for k, v in tag_weight_tuple_tf.items()]

	return actor_weight_vector_tf
	
def userMovieRatings(user_id):
	movieRatings = dict()
	count = 0
	sum = 0
	for row in user_ratings_df.itertuples():
		if(row.userid == user_id):
			rating = row.rating
			count += 1
			sum += rating
			movieRatings[row.movieid] = rating

	mean = float(sum)/float(count)
	for row in tag_movie_df.itertuples():
		if(row.userid == user_id):
			if(row.movieid not in movieRatings):
				movieRatings[row.movieid] = mean
				sum += mean

	return [(k,v/sum) for k,v in movieRatings.items()]
