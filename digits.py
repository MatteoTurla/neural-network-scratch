#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 11:53:01 2020

@author: matteoturla
"""
from matplotlib import pyplot as plt
from project import ComputationlGraph, Dense, Sigmoid, SoftmaxCrossEntropy
import numpy as np
import time

# read saved data
Xtrain = np.genfromtxt('data/Xtrain.csv', delimiter=',')
Xtest = np.genfromtxt('data/Xtest.csv', delimiter=',')
Ytrain = np.genfromtxt('data/Ytrain.csv', delimiter=',')
Ytest = np.genfromtxt('data/Ytest.csv', delimiter=',')


# init the model
model = ComputationlGraph([Dense(64,128), Sigmoid(), 
                           Dense(128,128), Sigmoid(), 
                           Dense(128,64), Sigmoid(), 
                          Dense(64,10)], SoftmaxCrossEntropy())

# train model
start = time.time()

loss_iter = []
max_iter = 1000
for i in range(0,max_iter):
    output, cost = model.forward(Xtrain,Ytrain)
    loss_iter.append(cost)
    model.backward()
    model.update(0.0001)

end = time.time()
print("time elpased: ", end - start)

# plot loss function
plt.figure()
plt.plot(range(0,max_iter), loss_iter, 'o');
plt.show()


# evalutee the model
output, cost = model.forward(Xtest, Ytest)
output_to_class = np.argmax(output, axis=0) 
output_to_class_true = np.argmax(Ytest, axis=0) 
acc = (output_to_class == output_to_class_true).mean()
print("accuarcy: ",acc)


# SGD

# init the model
model = ComputationlGraph([Dense(64,128), Sigmoid(), 
                           Dense(128,128), Sigmoid(), 
                           Dense(128,64), Sigmoid(), 
                          Dense(64,10)], SoftmaxCrossEntropy())

print("\n \n \nSGD")
epochs = 1000
batch = 256
n_examples = Xtrain.shape[1]
print("number of examples: ", n_examples)
print("batch number: ", batch)

start = time.time()

loss_epochs = []
for e in range(0, epochs):
    # shuffle the data
    Xshuffle = Xtrain.T.copy()
    Yshuffle = Ytrain.T.copy()
    index = np.random.permutation(Xshuffle.shape[0])
    loss_batch = 0
    current_batch = 0
    # train all over the entire data
    while current_batch < n_examples:
        if current_batch + batch  < n_examples:
            batch_row_index = index[current_batch:current_batch + batch]
            Xbatch = Xshuffle[batch_row_index,:]
            Ybatch = Yshuffle[batch_row_index, :]
            current_batch = current_batch + batch
        else:
            batch_row_index = index[current_batch:]
            Xbatch = Xshuffle[batch_row_index,:]
            Ybatch = Yshuffle[batch_row_index, :]
            current_batch = current_batch + batch
            
        output, cost = model.forward(Xbatch.T,Ybatch.T)
        loss_batch += cost
        model.backward()
        model.update(0.0001)
    loss_epochs.append(loss_batch)

end = time.time()
print("time elpased: ", end - start)

# evalutee the model
output, cost = model.forward(Xtest, Ytest)
output_to_class = np.argmax(output, axis=0) 
output_to_class_true = np.argmax(Ytest, axis=0) 
acc = (output_to_class == output_to_class_true).mean()
print("accuarcy: ",acc)


# plot loss function
plt.figure()
plt.plot(range(0,epochs), loss_epochs, 'o');
plt.show()
    

