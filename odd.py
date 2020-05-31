#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 11:58:49 2020

@author: matteoturla
"""

from matplotlib import pyplot as plt
from project import ComputationlGraph, Dense, Sigmoid, SigmoidBinaryCrossEntropyStable
from project import plot_decision_region
import numpy as np


X = np.arange(0,100)
Y = [x%3 for x in X]
Y = np.array(Y)

Y = np.expand_dims(Y, axis = 0)

BIN = [np.binary_repr(x, width=8) for x in X]
X = np.array([list(s) for s in BIN], dtype=np.float64).T




hidden1 = Dense(8, 16)
activation1 = Sigmoid()

hidden2 = Dense(16,32)
activation2 = Sigmoid()

hidden3 = Dense(32,32)
activation3 = Sigmoid()

hidden4 = Dense(32,16)
activation4 = Sigmoid()

hidden5 = Dense(16,1)
cost = SigmoidBinaryCrossEntropyStable()

model = ComputationlGraph([hidden1, activation1, 
                           hidden2, activation2, 
                           hidden3, activation3,
                           hidden4, activation4,
                           hidden5
                           ], cost)


loss_iter = []
max_iter = 1000
for i in range(0,max_iter):
    output, cost = model.forward(X,Y)
    loss_iter.append(cost)
    model.backward()
    model.update(1)

print(output > 0.5)
print(output)
print(Y)

acc = ((output >0.5) == Y).mean()
print("accuarcy: ",acc)

plt.figure()
plt.plot(range(0,max_iter), loss_iter, 'o');
plt.show()
