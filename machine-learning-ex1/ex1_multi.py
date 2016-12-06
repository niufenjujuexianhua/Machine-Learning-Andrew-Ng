# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 10:32:28 2016

@author: twu
"""
import numpy as np
import matplotlib.pyplot as plt

def trainingData(data):
    X = data[:,:-1]
    y = data[:,-1:]
    return X, y

def featureNormalize(X):
    xmu = np.average(X, axis=0)
    std = np.std(X, axis=0)
    norm = (X - xmu)/std 
    return norm
    
def computeCostMulti(X, y, theta):
    J = 0.5*np.sum(np.dot(X, theta) - y)/X.shape[0]
    return J
    
def gradientDescentMulti(X, y, theta, alpha, num_iters):
    J_history = np.zeros((num_iters, 1))
    for i in range(num_iters):
        gradient = np.dot(X.T,(np.dot(X, theta) - y))/X.shape[0]
        J = computeCostMulti(X, y, theta)
        J_history[i,:] = J
        theta = theta - alpha*gradient
    return J_history
    
def visualizeLearning(iter, J_history):
    iter = range(iter)
    plt.plot(iter, J_history)
    plt.show()
    
    
file = r"C:\Users\twu\Documents\Python Scripts\Machine Learning Coursera\Week1\ex1\ex1data2.txt"
data = np.loadtxt(file, delimiter=',')
alpha = 0.01
num_iters = 500
theta = np.zeros((3, 1))

X, y = trainingData(data)

Xnorm = featureNormalize(X)

Xnorm = np.c_[np.ones((Xnorm.shape[0], 1)), Xnorm]

J_history = gradientDescentMulti(Xnorm, y, theta, alpha, num_iters)

visualizeLearning(num_iters, J_history)            
              
              
              
              
              
              
              