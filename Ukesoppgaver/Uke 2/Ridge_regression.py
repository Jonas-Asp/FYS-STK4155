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

m = 100
x = np.random.rand(m,1)
y = 5*x*x+0.1*np.random.randn(m,1)
# Setting different lambda values
lam = [0.1,2,20,100,1000]
# Setting the lower and upper x-boundry and choosing number of points
xnew = np.linspace(0,1,m)

# Setting up ypredict matrix
ypredict = np.zeros((m,5))

# Going through the ridge regrissin for different lambda values
for i in range(5):
    ypredict[:,i] = ridge_regression(x,xnew,y,m,lam[i])
    plt.plot(xnew, ypredict[:,i], label=lam[i])
    
    print("Lambda paramater = %.1f" %lam[i])
    # The mean squared error                               
    print("Mean squared error = %.2f" %mean_squared_error(y, ypredict[:,i]))

    # R2 score. Negative because the fit is worse than a horizontal line
    print("R2 score = %.2f" % r2_score(y, ypredict[:,i]))
    print("")

plt.plot(x, y ,'ro')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.legend()
plt.title(r'Random numbers ')
plt.show()


# For large lambda variables low noise the coefficents are reduced to almost zero. While the smaller coenside more with OLS
# With increasing noise the lambda dependence reduces and the regression tends toward eachother.