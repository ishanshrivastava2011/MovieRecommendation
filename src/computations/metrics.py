from util import constants
from gensim.matutils import kullback_leibler
import math

def feature_combine(vec1, vec2):
	return set.intersection(set(vec1.keys()), set(vec2.keys()))

def euclidean(vec1,vec2):
	vec1 = dict(vec1)
	vec2 = dict(vec2)
	shared_features = feature_combine(vec1, vec2)
	return sum(list(map(lambda x: (vec2.get(x,0.0) + vec1.get(x,0.0)),shared_features)))

def simlarity_kullback_leibler(lda_vec1,lda_vec2):
    return kullback_leibler(lda_vec1,lda_vec2)


def euclideanDistance(vec1,vec2):
    union_feature = set.union(set(vec1.keys()), set(vec2.keys()))
    return sum(list(map(lambda x: (vec2.get(x,0.0) - vec1.get(x,0.0))**2,union_feature)))**0.5

'''
Takes in two vectors as lists and assumes the length of the vectors is the same.
returns cosine similarity between the two vectors as a float value.
'''
def cosineSim(vec1, vec2):
    cos = 0.0
    magv1 = 0.0
    magv2 = 0.0
    for i in range(0, len(vec1)):
        cos = cos + vec1[i]*vec2[i]
        magv1 = magv1 + vec1[i]*vec1[i]
        magv2 = magv2 + vec2[i]*vec2[i]
    return cos/(math.sqrt(magv1*magv2))

'''
Takes in two vectors as lists and assumes the length of the vectors is the same.
returns the l2 Norm between the two vectors.
'''
def l2Norm(vec1, vec2):
    product = 0
    for i in range(0, len(vec1)):
        product = product + (vec1[i] - vec2[i])*(vec1[i] - vec2[i])
    return math.sqrt(product)