# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 11:31:28 2016

@author: twu
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


#Visualize data
def visualizeData(X, y):
    plt.scatter(X, y)
    plt.xlabel('Population')
    plt.ylabel('Profit')
    plt.xlim(4, 24)
    plt.ylim(-5, 25)
    plt.show()

#Visualize learning curve
def visualizeLearning(iter, cost):
    plt.plot(iter, cost)
    plt.show()
    
    
#plot cost against theta in a 3D plot
def visualizeCostPlane(theta_hisotry, X, y):
    a = theta_history[:,0]
    b = theta_history[:,1]
    A, B = np.meshgrid(a, b)
    J_val = np.zeros((len(a), len(b)))

    for i in range(len(a)):
        for j in range(len(b)):
            theta = np.array( [ [a[i]], [b[j]] ])
            J_val[i,j] = computeCost(X, y, theta)

    J_val = J_val.T
    fig = plt.figure()
    ax = fig.gca(projection = '3d')
     
    surf = ax.plot_surface(A, B, J_val, 
                          rstride = 3,
                          cstride = 3,
                          cmap = cm.coolwarm,
                          linewidth = 0.5,
                          antialiased = True)
    
    fig.colorbar(surf, 
                 shrink=0.8, 
                 aspect=16,
                 orientation = 'vertical')
    
    ax.view_init(elev=60, azim=50)
    ax.dist=8 
    plt.show()
    
    
#prepare training data  
def trainingdata(array):
    X = array[:,0]
    X = np.c_[np.ones(X.shape[0]), X]
    y = array[:,1:]
    return X, y

#Compute cost   
def computeCost(X, y, theta):
    J = 0.5*sum(np.square(np.dot(X, theta) - y))/X.shape[0]
    return J
    
#perform gradient descent
def gradientDescent(X, y, theta, alpha, iterations):
    J_history = np.zeros((iterations, 1))
    theta_history = np.zeros((iterations, 2))
    for i in range(iterations):
        gradient = np.dot(X.T,(np.dot(X, theta) - y))/X.shape[0]
        theta = theta - alpha*gradient
        J = computeCost(X, y, theta)
        J_history[i,0] = J
        theta_history[i,0] = theta[0,0]
        theta_history[i,1] = theta[1,0]
    return theta_history, theta, J_history
        
    
    
file = r"C:\Users\twu\Documents\Python Scripts\Machine Learning Coursera\Week1\ex1\ex1data1.txt"
data = np.loadtxt(file, delimiter=',')

X, y = trainingdata(data)
theta = np.zeros((2, 1))
iterations = 1500
alpha = 0.01

theta_history, theta, J_history = gradientDescent(X, y, theta, alpha, iterations)

visualizeData(X[:,1], y)

visualizeLearning(range(iterations), J_history)

visualizeCostPlane(theta_history, X, y)



