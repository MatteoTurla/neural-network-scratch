#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 11:22:18 2020

@author: matteoturla
"""

from project import ComputationlGraph, Dense, Sigmoid, SigmoidBinaryCrossEntropyStable
from project import plot_decision_region
import numpy as np

### XOR example

X = np.array([[0,0],[0,1],[1,0],[1,1]]).T
Y = np.array([[0,1,1,0]])

hidden1 = Dense(2, 3)
activation1 = Sigmoid()

hidden2 = Dense(3,1)

cost = SigmoidBinaryCrossEntropyStable()

graph = ComputationlGraph([hidden1, activation1, hidden2], cost)


for i in range(0, 1000):
    output, cost = graph.forward(X,Y)
    graph.backward()
    graph.update(3)

print(output > 0.5)
print(Y)



plot_decision_region(X,Y, graph)