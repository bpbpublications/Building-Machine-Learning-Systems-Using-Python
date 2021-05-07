#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
def sigmoidfun(x):
    return 1.0/(1.0 + np.exp(-x))
def sigmoid_primefun(x):
    return sigmoidfun(x)*(1.0-sigmoidfun(x))
def tanh(x):
    return np.tanh(x)
def tanh_prime(x):
    return 1.0 - x**2


class NeuralNetwork:
    def __init__(self, layers, activation='tanh'):
        if activation == 'sigmoid':
            self.activation = sigmoidfun
            self.activation_prime = sigmoid_primefun
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_prime
        # Setting weights
        self.weights = []
        # let layers is [2,2,1]
        # weight values range= (-1,1)
        # hidden and input layers - random((2+1, 2+1)) : 3 x 3
        for i in range(1, len(layers) - 1):
            r = 2*np.random.random((layers[i-1] + 1, layers[i] + 1)) -1
            self.weights.append(r)
        # output layer - random
        r = 2*np.random.random( (layers[i] + 1, layers[i+1])) - 1
        self.weights.append(r)

    def fit(self, P, q, learning_rate=0.2, epochs=100000):
        # Adding column of ones to P
        # This is to add the bias unit to the input layer
        ones = np.atleast_2d(np.ones(P.shape[0]))
        P = np.concatenate((ones.T, P), axis=1)
        for k in range(epochs):
            i = np.random.randint(P.shape[0])
            a = [P[i]]
            for l in range(len(self.weights)):
                    dot_value = np.dot(a[l], self.weights[l])
                    activation = self.activation(dot_value)
                    a.append(activation)
            # output layer
            error = q[i] - a[-1]
            deltas = [error * self.activation_prime(a[-1])]
            # we need to begin at the second to last layer 
            # (a layer before the output layer)
            for l in range(len(a) - 2, 0, -1): 
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_prime(a[l]))

            # reverse
            # [level3(output)->level2(hidden)]  => [level2(hidden)->level3(output)]
            deltas.reverse()

            # backpropagation
            # 1. Multiply its output delta and input activation 
            #    to get the gradient of the weight.
            # 2. Subtract a ratio (percentage) of the gradient from the weight.
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)
            if k % 10000 == 0: print('epochs:', k)
    def predict(self, x): 
        c = np.concatenate((np.ones(1).T, np.array(x)))      
        for l in range(0, len(self.weights)):
            c = self.activation(np.dot(c, self.weights[l]))
        return c

if __name__ == '__main__':
    neu = NeuralNetwork([2,2,1])
    P = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    q= np.array([0, 1, 1, 0])
    neu.fit(P,q)
    for x in P:
        print(x,neu.predict(x))


# In[ ]:




