#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
 
class Perceptronlearning(object):
    """Implementation of  a perceptron learning network"""
    def __init__(self, input_size):
        self.W = np.zeros(input_size+1)
def activation_fn(self, p):
#“”” 1 is returned if p>=0 otherwise it returns 0”””
    return 1 if p >= 0 else 0
#“”” Prediction is a process of sending an input to the perceptron and returning an output. Bias is added to the input vector. We can compute inner product and activation function is applied ”””
def predict(self, p):
    p = np.insert(p, 0, 1)
    q = self.W.T.dot(p)
    r = self.activation_fn(q)
    return r
def __init__(self, input_size, lr=1, epochs=10):
    self.W = np.zeros(input_size+1)
    # add one for bias
    self.epochs = epochs
    self.lr = lr
def fit(self, P, d):
    for _ in range(self.epochs):
        for i in range(d.shape[0]):
            y = self.predict(P[i])
            e = d[i] - y
            self.W = self.W + self.lr * e * np.insert(P[i], 0, 1)
class Perceptronlearning(object):
    """Implements a perceptron network"""
    def __init__(self, input_size, lr=1, epochs=100):
        self.W = np.zeros(input_size+1)
        # add one for bias
        self.epochs = epochs
        self.lr = lr
    
    def activation_fn(self, p):
        #return (p >= 0).astype(np.float32)
        return 1 if p >= 0 else 0
 
    def predict(self, p):
        q = self.W.T.dot(p)
        r = self.activation_fn(q)
        return r
 
    def fit(self, P, d):
        for _ in range(self.epochs):
            for i in range(d.shape[0]):
                p = np.insert(P[i], 0, 1)
                y = self.predict(p)
                e = d[i] - y
                self.W = self.W + self.lr * e * p
if __name__ == '__main__':
    P = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    d = np.array([1, 1, 0, 1])
 
    perceptron = Perceptronlearning(input_size=2)
    perceptron.fit(P, d)
    print(perceptron.W)


# In[ ]:




