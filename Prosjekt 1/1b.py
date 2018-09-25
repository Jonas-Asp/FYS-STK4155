#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 10:33:32 2018

@author: Jonas Asperud
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from scipy import linalg
from sklearn.preprocessing import PolynomialFeatures

 # Bootstrap
def bootstrap(sampleData,nBoots,designMatrix,lam,shape):
    # Initiate matrices
    bootVec = np.zeros(len(sampleData))
    b = np.zeros((nBoots,designMatrix.shape[1]))
    mse = np.zeros(nBoots)
    r2 = np.zeros(nBoots)
    # Assigns the last third as test data and the first 3 thirds as training data
    threeThirds = int(3*len(sampleData)/4)
    trainingData = sampleData[0:threeThirds]
    testData = sampleData[threeThirds+1:]
    # Loops through the set number of boots
    for k in range(0,nBoots):
        # Chooses a random set of data points from data, with same number of rows as data
        bootVec = np.random.choice(trainingData, len(sampleData))
        # Calculates beta values
        b[k,:] = linalg.inv(designMatrix.T.dot(designMatrix) - lam*np.identity(shape)).dot(designMatrix.T).dot(bootVec)
        # Create a fit model for the data
        bootpred = designMatrix.dot(b[k,:]).flatten()
        # Does error calculation
        mse[k],r2[k] = error(testData,bootpred[threeThirds+1:])
    return np.mean(mse),np.mean(r2),b

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

# Calculates beta values and predict the function using this fit/model
def regression(z,designMatrix,lam,shape):
    beta = np.zeros((len(lam),shape))
    for k in range(len(lam)):
        beta[k] = linalg.inv(designMatrix.T.dot(designMatrix)-lam[k]*np.identity(shape)).dot(designMatrix.T).dot(z)
    return beta


# Error evaluation
def error(z,zpred):
    mse = (1/len(z))*sum((z-zpred)**2)
    mean = (1/len(z))*sum(z)
    r2 = 1 - (len(z)*mse/sum((z-mean)**2))
    return mse,r2
    
# Make data.
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)
z = FrankeFunction(x,y)

# Setting the right format for the matrices and add noise
Z = z.flatten()+0*np.random.rand(z.shape[0]*z.shape[1],1).flatten()
X = x.flatten()
Y = y.flatten()

# Creating design matrix
XY = np.c_[X,Y]
degree = PolynomialFeatures(degree=5)
dMatrix = degree.fit_transform(XY)

#Setting lambda value. For ridge lam > 0; for OLS lam = 0
lam = [0,1,2,3,4,5,6,7,8,9,10]
beta = regression(Z,dMatrix,lam,21)










bootMSE = np.zeros(len(lam))
bootR2 = np.zeros(len(lam))
MSE = np.zeros(len(lam))
R2 = np.zeros(len(lam))
VAR = np.zeros(len(lam))
# Prediction and plotting
for i in range(len(lam)):
    zpred = dMatrix.dot(beta[i,:].flatten()).flatten()
    
    # Error and variance etc.
    MSE[i],R2[i] = error(Z,zpred)
    # Bootstrapping
    bootMSE[i], bootR2[i], bootBeta = bootstrap(Z,1000,dMatrix,lam[i],dMatrix.shape[1])
    
    
    sigma = (5+1)*MSE[i]
    varB = linalg.inv(dMatrix.T.dot(dMatrix))*(sigma)
    confInter = 2*np.sqrt(np.diagonal(varB))
    

# Bootstrap MSE plot
plt.plot(lam,bootMSE,'go-',label=('Minimum MSE = %.4f at $\lambda$=%.0f' %(MSE[np.argmin(MSE)],lam[np.argmin(MSE)])))
plt.title('Ridge regression - Mean bootstrap MSE for different lambda values')
plt.xlabel('Lambda value')
plt.ylabel('Mean bootstrapping MSE')
plt.legend()
plt.show()

# Mean square error plot
plt.plot(lam,MSE,'ro-',label='Ridge')
plt.plot(lam,np.ones(len(lam))*MSE[0],'g--',label='OLS')
plt.plot(lam[np.argmin(MSE)],MSE[np.argmin(MSE)],'bo',label=('Minimum MSE = %.4f at $\lambda$=%.0f' %(MSE[np.argmin(MSE)],lam[np.argmin(MSE)])))
plt.xlabel('Lambda value')
plt.ylabel('Mean square error (MSE)')
plt.title('Ridge regression - Lambda value as a function of mean square error')
plt.legend()
plt.show()

# Bootstrap MSE plot
bestR2 = (np.abs(bootR2-1)).argmin()
plt.plot(lam,bootR2,'go-',label=('Best R2 score = %.4f at $\lambda$=%.0f' %(bootR2[bestR2],lam[bestR2])))
plt.title('Ridge regression - Mean bootstrap R2 score for different lambda values')
plt.xlabel('Lambda value')
plt.ylabel('Mean bootstrapping R2 score')
plt.legend()
plt.show()

# R2 score plot
bestR2 = (np.abs(R2-1)).argmin()
plt.plot(lam,R2,'ro-',label='Ridge')
plt.plot(lam,np.ones(len(lam))*R2[0],'g--',label='OLS')
plt.plot(lam[bestR2],R2[bestR2],'bo',label=('Best R2 score = %.4f at $\lambda$=%.0f' %(R2[bestR2],lam[bestR2])))
plt.xlabel('Lambda value')
plt.ylabel('R2 score')
plt.title('Ridge regression - Lambda values as a function of R2 score')
plt.legend()
plt.show()



