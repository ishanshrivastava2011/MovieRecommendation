import tensorflow as tf
import numpy as np


class BinarySVM:

    def __init__(self, session=tf.Session(), func='linear', lr=0.9, iterations=1000, scope='scope'):
        self.funcs = {'linear' : self.linearKernel}
        self.func = func
        self.session = session
        self.loss = -1
        self.iterations = iterations
        self.lr = lr
        self.test_vars = None
        self.scope = scope
        self.model = None
        self.bias = None
        self.weights = None
        self.f_w = None
        self.f_b = None


    def initVariableBias(self, num_vars):
        # with tf.variable_scope(self.scope):
        self.bias = tf.Variable(tf.random_normal([1]))
        self.weights = tf.Variable(tf.random_normal([num_vars, 1]))

    def linearKernel(self, input_x):
        return tf.tensordot(input_x, self.weights, 1) + self.bias


    def fit(self, X, y):
        num_vars = X.shape[1]
        X_data = tf.constant(X, dtype=tf.float32)
        label_y = tf.constant(y, dtype=tf.float32)
        self.initVariableBias(num_vars)
        self.model = self.linearKernel(X_data)
        loss = tf.reduce_mean(tf.losses.hinge_loss(label_y, self.model))
        train_step = tf.train.AdamOptimizer(0.9).minimize(loss)
        self.test_vars = tf.placeholder(tf.float32, [None, num_vars])
        self.session.run(tf.global_variables_initializer())

        for i in range(1000):
            cost,_ = self.session.run(loss), self.session.run(train_step)
            # print(cost)

        self.f_w = self.session.run(self.weights)
        self.f_b = self.session.run(self.bias)
        m = self.session.run(self.model)

        return self.loss


    def predict(self, X):
        return np.dot(X,np.array(self.f_w)) + self.f_b
