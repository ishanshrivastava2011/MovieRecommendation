import tensorflow as tf
import numpy as np
from Classifiers.TSVM import BinarySVM as bsvm
from sklearn import datasets
import time


class kSVM:

    def __init__(self, func='linear', lr=0.9, iterations=1000):
        self.func = func
        self.session = tf.Session()
        self.loss = -1
        self.iterations = iterations
        self.lr = lr
        self.model = []
        self.X = None

    def __onehottransform(self, data, encoder):
        for i, d in enumerate(data):
            encoder[int(d)][i] = 1.0
        return

    def __onehotencoder(self, data):
        ids = set(data).__len__()
        encoder = np.ones((ids, len(data)), dtype=np.float32) * -1
        self.__onehottransform(data, encoder)
        return encoder

    def __fitModel(self, X, arr, tf_session):
        new_X = X
        narr = arr
        svm = bsvm(session=tf_session,lr=self.lr, iterations=self.iterations, func=self.func )
        svm.fit(X, narr.reshape([narr.shape[0], 1]))
        m = svm.predict(X)
        return svm

    def fit(self, X, y):


        encoders = self.__onehotencoder(y)
        sess = tf.Session()
        for encoder in encoders:
            self.model.append(self.__fitModel(X,encoder, sess))



    def predict(self, X):
        m = []
        for model in self.model:
            m.append(model.predict(X))

        predictions = np.array(m)
        predictions = predictions.reshape((predictions.shape[0],predictions.shape[1])).T
        return np.argmax(predictions, axis=1) + 1


if __name__ == "__main__":
    iris = datasets.load_iris()
    start_time = time.time()
    X = iris.data.astype(dtype=np.float32)
    y = iris.target.astype(dtype=np.float32)
    d = kSVM()
    timer = time.time()
    d.fit(X,y)
    d.predict(X)
    print(time.time() - timer)
    print("Done!")



