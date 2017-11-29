from util import constants
from computations import decompositions
import pickle

#PCA
#PCA = decompositions.PCADimensionReduction((movie_tag_df), 455)
#PCA = pd.DataFrame(PCA)
#PCA.to_pickle(constants.DIRECTORY + "PCA_decomposition")
#movie_tag_df = pd.read_pickle(constants.DIRECTORY + "movie_tag_df.pickle")
#SVD
def create_SVD_pickle(movie_tag_df) :
    SVD_local = None
    try:
        SVD_local = pickle.load(open(constants.DIRECTORY + "SVD_decomposition.pickle", "rb"))
    except (OSError, IOError) as e:
        SVD_local = decompositions.SVDDecomposition((movie_tag_df), 2518)[0]
        pickle.dump(SVD_local, open(constants.DIRECTORY + "SVD_decomposition.pickle", "wb"))  
    
    return SVD_local

def create_PCA_pickle(movie_tag_df) :
    PCA_local = None
    try:
        PCA_local = pickle.load(open(constants.DIRECTORY + "PCA_decomposition.pickle", "rb"))
    except (OSError, IOError) as e:
        PCA_local = decompositions.SVDDecomposition((movie_tag_df), 1160)[0]
        pickle.dump(PCA_local, open(constants.DIRECTORY + "PCA_decomposition.pickle", "wb"))  
    
    return PCA_local

#MDS
#SVD_FOR_MDS = decompositions.SVDDecomposition((movie_tag_df), 500)[0]
#SVD_FOR_MDS = pd.DataFrame(SVD, index =movie_tag_df.index)
#SVD_FOR_MDS.to_csv(constants.DIRECTORY+'MoviesinLatentSpace_SVD_MDS.csv')