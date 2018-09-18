# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 15:25:01 2018

@author: Rafiki
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 10:26:33 2018

@author: Rafiki
"""

# Importing various packages
from random import random, seed
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

 # Bootstrap
def bootstrap(self, nBoots = 1000):
    bootVec = np.zeros(nBoots)
    for k in range(0,nBoots):
        bootVec[k] = np.average(np.random.choice(self.data, len(self.data)))
    bootAvg = np.average(bootVec)
    bootVar = np.var(bootVec)
    plt.hist(bootVec, bins='auto')
    plt.show()
    return bootAvg,bootVar

# FrankeFuncton
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


# Finner beta verdier
xb = np.c_[np.ones((len(z),1)), x,x**2,x**3,x**4,x**5]
beta = np.linalg.inv(xb.T.dot(xb)).dot(xb.T).dot(z)
zpred = beta[0]+(beta[1]*x)+(beta[2]*x**2)+(beta[3]*x**3)+(beta[4]*x**4)+(beta[5]*x**5)

# The beta values as polynomial
print("######## Beta verdier i stigende rekkefølge ###########")
print(beta)      
print("")

# Ser på error
mse = mean_squared_error(z,zpred)
print("######## Error vurdering ###########")
print("MSE = %.2f" %mse)
print("R2 = %.2f" %r2_score(z,zpred)) 
print("")     

# Finner variansen som funksjon av covariance matrisen
sigma = mse*(100/(100-5-1))
var = np.linalg.inv(xb.T.dot(xb)).dot(sigma)
print("######## Variansen som diagonal elementer i covarians matrisen ###########")
print(np.diagonal(var))
print("")     

# Confidense interval ?
print("######## Confidence interval som 2*sqrt(diagonal(var)) ??? ###########")
print(2*np.sqrt(np.diagonal(var)))

# Resampling
resamp = bootstrap(zpred.flatten())
print("")
print("######## Histogram av bootstrap gjennomsnitt ###########")  
print("Bootstrap gjennomsnitt = %.2f og varians = %.2f" %(resamp[0],resamp[1]))
print("Sann verdi gjennomsnitt = %.2f og varians = %.2f" %(np.mean(z),np.var(z)))

xnew = np.linspace(0,1,100)
ypred = beta[0]+(beta[1]*xnew)+(beta[2]*xnew**2)+(beta[3]*xnew**3)+(beta[4]*xnew**4)+(beta[5]*xnew**5)
plt.plot(x,z,'ro')
plt.plot(xnew,ypred)
plt.show()