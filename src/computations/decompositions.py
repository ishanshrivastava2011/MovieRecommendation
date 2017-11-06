from sklearn.utils.extmath import randomized_svd
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
import pandas as pd
import util.constants
import gensim
from gensim import corpora
import numpy as np
from sktensor import dtensor, cp_als

def PCADecomposition(inputMatrix, n_components):
	mat_std = StandardScaler().fit_transform(inputMatrix,inputMatrix.columns.values)
	pca = decomposition.PCA(n_components)
	pca.fit(mat_std)
	return pca.components_


def PCADimensionReduction(inputMatrix,new_dimensions):
	mat_std = StandardScaler().fit_transform(inputMatrix, inputMatrix.columns.values)
	pca = decomposition.PCA(new_dimensions)
	pca.fit(mat_std)
	return pca.transform(mat_std)

"""
inputMatrix is assumed to be in the pandas format
n_componenets is the top latent semantics we need. The Sigma matrix will have these many values in its list
Sigma is a list of values and not a matrix
"""
def SVDDecomposition(inputMatrix, n_components):
    mat_std = StandardScaler().fit_transform(inputMatrix,inputMatrix.columns.values)
    U, Sigma, VT = randomized_svd(mat_std,
                              n_components,
                              n_iter=util.constants.ITERATIONS,
                              random_state=util.constants.RANDOM_STATE)
    return U, Sigma, VT

def LDADecomposition(inputMatrix, num_topics, passes):
    #Removing all the zero columns
    df = inputMatrix.loc[:, (inputMatrix != 0).any(axis=0)]
    
    doc_term_matrix1 = df.as_matrix()
    df1 = pd.DataFrame(df.columns)
    id_Term_map = dict(zip(df1.index, df.columns))
    numpy_matrix = np.matrix(doc_term_matrix1)
    numpy_matrix_transpose = numpy_matrix.transpose()
    
    corpus = gensim.matutils.Dense2Corpus(numpy_matrix_transpose)
    dictionaryFromCorpus = corpora.Dictionary.from_corpus(corpus)
    SOME_FIXED_SEED = 42

    # before training/inference:
    np.random.seed(SOME_FIXED_SEED)
    Lda = gensim.models.ldamodel.LdaModel
    ldamodel = Lda(corpus, num_topics=num_topics, id2word = dictionaryFromCorpus, passes=passes, random_state=1)
    
    return ldamodel,corpus,id_Term_map

def CPDecomposition(tensor,rank):
    T = dtensor(tensor)
    # Decompose tensor using CP-ALS
    P, fit, itr, exectimes = cp_als(T, rank, init='random')
    u=P.U
    return u