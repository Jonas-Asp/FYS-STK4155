#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 10:33:32 2018

@author: Jonas Asperud
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from scipy import linalg
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import PolynomialFeatures

 # Bootstrap
def bootstrap(sampleData,nBoots,designMatrix):
    # Initiate matrices
    bootVec = np.zeros(len(sampleData))
    b = np.zeros((nBoots,designMatrix.shape[1]))
    mse = np.zeros(nBoots)
    r2 = np.zeros(nBoots)
    
    # Loops through the set number of boots
    for k in range(0,nBoots):
        # Chooses a random set of data points from data, with same number of rows as data
        bootVec = np.random.choice(sampleData, len(sampleData))
        # Calculates beta values
        b[k,:] = linalg.inv(designMatrix.T.dot(designMatrix)).dot(designMatrix.T).dot(bootVec)
        # Create a fit model for the data
        bootpred = designMatrix.dot(b[k,:]).flatten()
        # Does error calculation
        mse[k] = mean_squared_error(sampleData,bootpred)
        r2[k] = r2_score(sampleData,bootpred)
    return mse,r2,b

# Creating the FrankeFunction
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

#Bare for ha en enkel plotte funksjon
def plotting(x,y,z):
    # Plot the surface.
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x,y,z,cmap=plt.cm.viridis,linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()    

# Make data.
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)
z = FrankeFunction(x,y)

# Setting the right format for the matrices
Z = z.flatten()
X = x.flatten()
Y = y.flatten()

# Creating design matrix
XY = np.c_[X,Y]
degree = PolynomialFeatures(degree=5)
dMatrix = degree.fit_transform(XY)

# Calculates beta values and predict the function using this fit/model
beta = linalg.inv(dMatrix.T.dot(dMatrix)).dot(dMatrix.T).dot(Z)

# Bootstrapping (The real matrix, Number of boots, design-matrix)
bootMSE, bootR2, bootBeta = bootstrap(Z,1,dMatrix)


# Prediction and plotting
zpred = dMatrix.dot(bootBeta.flatten()).flatten()
zpred = zpred.reshape(20,20)
plotting(x,y,z)
plotting(x,y,zpred)









