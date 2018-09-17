# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 10:52:17 2018

@author: Rafiki

Ridge regression
"""

# Defining ridge regression algorithm
def ridge_regression(x,xnew,y,m,lam):
    # Set first column to ones and find beta
    xb = np.c_[np.ones((m,1)), x]
    
    beta = (np.linalg.inv(xb.T.dot(xb)+lam*np.identity(2)).dot(xb.T).dot(y))
    #print(np.linalg.inv(xb.T.dot(xb)+lam*np.identity(2)))
    
    
    # Creating the matrix with ones in the first column
    xbnew = np.c_[np.ones((m,1)),xnew]
    
    # Creating the matrix with m points between the boundries of xnew with the beta-model fit
    ypredict = xbnew.dot(beta)
    
    #Weird error where ypredict is m,1 matrix and has to return m matrix. Apperently not same
    return ypredict[:,0]


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model

m = 100
x = np.random.rand(m,1)
y = 5*x*x+0.1*np.random.randn(m,1)
# Setting different lambda values
lam = [0.1,2.0,20.0,100,1000]
# Setting the lower and upper x-boundry and choosing number of points
xnew = np.linspace(0,1,m)

# Setting up ypredict matrix
ypredict = np.zeros((m,3))
ypred = np.zeros((m,3))

# Going through the ridge regrissin for different lambda values
for i in range(3):
    ypredict[:,i] = ridge_regression(x,xnew,y,m,lam[i])
    plt.plot(xnew, ypredict[:,i], label=lam[i])
    
    print("---------")
    print("Lambda paramater = %.1f" %lam[i])
    print("---------")
    

    
    
    
    ridge=linear_model.RidgeCV(alphas=[lam[i]])
    ridge.fit(x,y)
    take = ridge.predict(x)    
    ypred[:,i] = take[:,0]
    plt.plot(x,ypred[:,i],'-',label=lam[i])
    
    # The mean squared error                               
    print("Mine - Mean squared error = %.2f" %mean_squared_error(y, ypredict[:,i]))
    print("Scikit - Mean squared error = %.2f" %mean_squared_error(y, ypred[:,i]))
    # R2 score. Negative because the fit is worse than a horizontal line
    print("Mine - R2 score = %.2f" % r2_score(y, ypredict[:,i]))
    print("Scikit - R2 score = %.2f" % r2_score(y, ypred[:,i]))
    print("")


plt.plot(x, y ,'ro')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.legend()
plt.title(r'Random numbers ')
plt.show()

# One of the differences in the error calculation is that my code has a linearly spaced x, while the difference 
# in scikit learn is that the x'es are spaced where they were indexed from the real data

