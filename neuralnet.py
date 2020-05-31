#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 10:48:11 2020

@author: matteoturla
"""

import numpy as np
from matplotlib import pyplot as plt

class ComputationlGraph:
    def __init__(self, listGate, costGate):
        # listGate is a list of Gate
        self.listGate = listGate
        self.costGate = costGate

    def forward(self, X, Y):
        for gate in self.listGate:
            X = gate.forward(X)
        Y_ = X
        output, cost = self.costGate.forward(Y, Y_)
        return output, cost
    
    def predict(self, X):
        for gate in self.listGate:
            X = gate.forward(X)
        Y_ = X
        Y_ = self.costGate.predict(Y_)
        return Y_

    def backward(self):
        upstream = 1
        upstream = self.costGate.backward(upstream)
        for gate in self.listGate[::-1]:
            upstream = gate.backward(upstream)

    def update(self, learning_rate):
        for gate in self.listGate:
            gate.update(learning_rate)



# class Gate:
#   self.saveComputation -> variable for save the computation and use it in backward pass
#   def forward(input) -> compute output given the input and save the computation
#   def backward(upstream) -> compute the derivative using chain rule
#   def update() -> update the weight if needed
#   All of this is done in vectorize computation, the input of the first layer is a matrix X of shape M*X,
#   where M is the number of Features and N is the number of examples in the batch
#   Each examples is a column vector, to obtain a batch of examples we stack them horizontally

class Dense:
    # the weight matrix W has shape M*N where M is the number of neurons in the current layer and N is the number 
    # of neurons in previous layer, in case of layer(0) N is the number of features.
    
    def __init__(self, previousNeurosLayer, currentNeuronsLayer):
        # number of neurons in the current layer
        self.currentNeuronsLayer = currentNeuronsLayer
        
        # initialize weights randomnly
        self.W = np.random.randn(currentNeuronsLayer, previousNeurosLayer)*np.sqrt(2/previousNeurosLayer)
        self.B = np.zeros((currentNeuronsLayer, 1))
        
        # save computations variables
        self.A = None
        self.dW = None
        self.dB = None

    def initWeights(self, W, B):
        # initialize weights defined by the user 
        self.W = W
        self.B = B

    def forward(self, A):
        # A is the output of the previous layer
        self.A = A
        Z = np.dot(self.W, self.A) + self.B
        return Z

    def backward(self, upstream):
        # compute local derivatives of the gate: dZ/dW, dZ/dB, dZ/dX
        localDW = self.A.T
        localDA = self.W.T
        
        # compute global derivatives: dL/dW, dL/dB, dL/dA
        self.dW = np.dot(upstream, localDW)
        self.dB = np.sum(upstream, axis=1, keepdims=True)
        dA = np.dot(localDA, upstream)
        
        return dA

    def update(self, learning_rate):
        # update the weights by gradient descent
        self.W = self.W  -1*learning_rate*self.dW
        self.B = self.B -1*learning_rate*self.dB

# ACTIVATION FUNCTIONS

class Sigmoid:
    # Sigmoid activation functions
    
    def __init__(self):
        # save computations variables
        self.A = None
        self.m = None

    def forward(self, Z):
        # Z is the output of the previous layer
        # compute the function for each element of matrix Z
        A = 1.0 / (1 + np.exp(-1*Z))
        self.A = A
        return A

    def backward(self, upstream):
        # compute the local gradient of the gate: dSigma/dZ
        local = self.A*(1-self.A)
        # compute global gradient: dL/dZ
        dZ = upstream*local
        return dZ

    def update(self, learning_rate):
        # nothing to update
        pass
    

class Relu:
    # Relu activation function
    
    def __init__(self):
        # save computations variables
        self.Z = None
        self.m = None

    def forward(self, Z):
        # Z is the output of the previous layer
        # compute the function for each element of matrix Z
        self.Z = Z
        A = Z.copy()
        A[A<0] = 0
        return A

    def backward(self, upstream):
        # compute the local gradient of the gate: dRelu/dZ
        local = 1. * (self.Z>0.)
        # compute global gradient: dL/dZ
        dZ = upstream*local
        return dZ

    def update(self, learning_rate):
        # notingh to update
        pass

class LeakyRelu:
    # Relu activation function
    
    def __init__(self):
        # save computations variables
        self.Z = None
        self.m = None

    def forward(self, Z):
        # Z is the output of the previous layer
        # compute the function for each element of matrix Z
        self.Z = Z
        A = np.where(Z > 0, Z, Z * 0.01)
        return A

    def backward(self, upstream):
        # compute the local gradient of the gate: dRelu/dZ
        local = np.where(self.Z > 0, 1, 0.01)
        # compute global gradient: dL/dZ
        dZ = upstream*local
        return dZ

    def update(self, learning_rate):
        # notingh to update
        pass


    
# LOSS FUNCTIONS

class MSE:
    # Minimum Square Error (L2 norm square)
    
    def __init__(self):
        self.Y_ = None
        self.Y = None
        self.m = None


    def forward(self, Y, Y_):
        self.Y_ = Y_
        self.Y = Y
        self.m = Y_.shape[1]
        diff = Y-Y_
        diffSquare = np.sum(np.square(diff), axis = 0)
        cost = 0.5*diffSquare.mean()
        return Y_, cost

    def predict(self, Y_):
        return Y_
    
    def backward(self, upstream):
        local = (-1/self.m)*(self.Y-self.Y_)
        dY_ = upstream*local
        #print(dY_)
        return dY_
    
class BinaryCrossEntropy:
    # Binary cross entropy loss, generally used for binary classification
    # L = -1/m * (sum_over_examples y*log(y_) + (1-y)*log(1-y_))
    # dL/dY_ = -1/m * (Y/Y_ + (1-y)/(1-y_)*(-1)) = 1/m * (-Y/Y_ + (1-Y)/(1-Y_))
    
    # this is numerically unstable due to the computation of the log of a number that can be near 0!
    
    def __init__(self):
        # computation save variables
        self.Y_ = None
        self.Y = None
        self.m = None

    def forward(self, Y, Y_):
        self.Y_ = Y_
        self.Y = Y
        self.m = Y_.shape[1]
        part1 = np.multiply(Y, np.log(Y_))
        part2 = np.multiply((1-Y), np.log(1-Y_))
        cost = -1/self.m*np.sum(part1+part2)
        return Y_, cost
    
    def predict(self, Y_):
        return Y_

    def backward(self, upstream):
        local = (1/self.m)*(-1*self.Y/self.Y_ + (1-self.Y)/(1-self.Y_))
        dY_ = upstream*local
        return dY_

    
class SigmoidBinaryCrossEntropyStable:
    # the loss is compute on Z instead of A, when we init this, 
    # we pass in the constructor the last activation layer
    
    def __init__(self):
        # computation save variables
        self.Y_ = None
        self.Y = None
        self.m = None

    def sigmoid(self, Z):
        # Z is the output of the previous layer
        # compute the function for each element of matrix Z
        A = 1.0 / (1 + np.exp(-1*Z))
        return A

    def forward(self, Y, Z):
        Y_ = self.sigmoid(Z)
        self.Y_ = Y_
        self.Y = Y
        self.m = Y_.shape[1]
        
        cost = (-1/self.m) * np.sum(-1*Y*np.log(1+np.exp(-1*Z)) + (1-Y)*(-1*Z-np.log(1+np.exp(-1*Z))))
        return Y_, cost

    def backward(self, upstream):
        local = (1/self.m)*(self.Y_ - self.Y)
        dY_ = upstream*local
        return dY_
    
    def predict(self, Z):
        return self.sigmoid(Z)
    
class SoftmaxCrossEntropy:
    def __init__(self):
        # computation save variables
        self.Y_ = None
        self.Y = None
        self.m = None

    def softmax(self, Z):
        maxA = np.max(Z, axis=0, keepdims=True)
        numerator = np.exp(Z-maxA)
        denominetor = np.sum(numerator, axis = 0, keepdims=True)
        output = numerator/denominetor
        self.A = output
        return output
    
    def forward(self, Y, Z):
        Y_ = self.softmax(Z)
        self.Y = Y
        self.Y_ = Y_
        self.m = Y_.shape[1]
        
        # compute loss
        cost = -1/self.m * np.sum(np.multiply(self.Y,np.log(self.Y_)))
        return Y_, cost
    
    def predict(self, Z):
        return self.softmaz(Z)

    def backward(self, upstream):
        local = self.Y_ - self.Y
        dY_ = upstream*local
        return dY_
    
    
# Plotting decision regions for 2D
def plot_decision_region(X,Y,model):
    # X matrix of M*N M number of features, N number of example
    x_min, x_max = X.T[:, 0].min() - 1, X.T[:, 0].max() + 1
    y_min, y_max = X.T[:, 1].min() - 1, X.T[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    
    surface_points = np.c_[xx.ravel(), yy.ravel()].T
    Z = model.predict(surface_points)
    
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X.T[:, 0], X.T[:, 1], c=Y[0],s=20, edgecolor='k')
    
    plt.show()
    
    