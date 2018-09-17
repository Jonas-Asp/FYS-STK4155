# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 08:24:22 2018

@author: Jonas Asperud
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

########## Make synthetic data ##########
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
z = FrankeFunction(x, y)




########## Ordinary least square regression ##########

m = x.size
xb = np.c_[np.ones((m,1)),x,y,x**2,y**2,x*y,x**3, x**2*y, x*y**2, y**3,x**4, x**3*y, x**2*y**2, x*y**3,y**4,x**5, x**4*y, x**3*y**2, x**2*y**3,x*y**4, y**5]
# Calculate the intercept and b1
beta = np.linalg.inv(xb.T.dot(xb)).dot(xb.T).dot(z)

xnew = np.linspace(0,1,20)    
ynew = np.linspace(0,1,20)

# Creating the matrix with m points between the boundries of xnew with the beta-model fit
zpredict = beta[0] + beta[1]*x + beta[2]*x + beta[3]*x**2 + beta[4]*y**2 + beta[5]*x*y + beta[6]*x**3 + beta[7]*x**2*y + beta[8]*x*y**2 + beta[9]*y**3 + beta[10]*x**4 + beta[11]*x**3*y + beta[12]*x**2*y**2 + beta[13]*x*y**3 + beta[14]*y**4 + beta[15]*x**5 + beta[16]*x**4*y + beta[17]*x**3*y**2 + beta[18]*x**2*y**3 + beta[19]*x*y**4 + beta[20]*y**5

# Fitting zpredict into a array with m points for MSE
zpred = np.linspace(zpredict[0],zpredict[1],m)
mse = mean_squared_error(z,zpred)
# Mean square error (Also an estimate of the variance, page 47)
#print('Mean squared error: %.2f' %mse)

# Variance
var = np.linalg.inv(xb.T.dot(xb))
#print('Variance of beta values')
#print(np.diagonal(np.linalg.inv(xb.T.dot(xb))))


xx = x.reshape(1,-1)
zz = z.reshape(1,-1)
linreg = LinearRegression()
linreg.fit(xx,zz)
xnew2 = np.linspace(0,1,20)
xnew2 = xnew2.reshape(1,-1)
ypredict = linreg.predict(xnew2)



## Plotting y because x=y and z is the function of the 2 predictors
plt.plot(y,z,'ro')
# Plotting the OLS
plt.plot(xnew, zpredict, "bo")
plt.plot(xnew2,ypredict,'go')
plt.xlabel(r'$y$')
plt.ylabel(r'$f(x,y)$')
plt.title('Ordinary least square fit for FrankeFunction')
plt.show()