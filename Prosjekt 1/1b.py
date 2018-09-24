#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 14:21:47 2018

@author: Rafiki
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from scipy import linalg
from sklearn.metrics import mean_squared_error,r2_score

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)

 # Bootstrap
def bootstrap(self, nBoots = 1):
    bootVec = np.zeros(nBoots)
    for k in range(0,nBoots):
        print(np.random.choice(self.data, len(self.data)))
    

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

z = FrankeFunction(x,y)
z1 = z.flatten()
x1 = x.flatten()
y1 = y.flatten()

lam = 0.01

# Finner beta verdier
xy = np.c_[np.ones((z.size,1)),x1,y1
           ,x1**2,x1*y1,y1**2
           ,x1**3,x1**2*y1,x1*y1**2,y1**3
           ,x1**4,x1**3*y1,x1**2*y1**2,x1*y1**3,y1**4
           ,x1**5,x1**4*y1,x1**3*y1**2,x1**2*y1**3,x1*y1**4,y1**5]

beta = linalg.inv(xy.T.dot(xy) - lam*np.identity(len(z)+1)).dot(xy.T).dot(z1)

zpred = xy.dot(beta).flatten()


# The beta values as polynomial
print("######## Beta verdier i stigende rekkefølge ###########")
print(beta)      
print("")

# Ser på error
mse = mean_squared_error(z1,zpred)
print("######## Error vurdering ###########")
print("MSE = %.4f" %mse)
print("R2 = %.4f" %r2_score(z1,zpred)) 
print("") 

# Finner variansen som funksjon av covariance matrisen
sigma = mse*(100/(100-5-1))
var = linalg.inv(xy.T.dot(xy)).dot(sigma)
print("######## Variansen som diagonal elementer i covarians matrisen ###########")
print(np.diagonal(var))
print("")  

# Confidense interval ?
print("######## Confidence interval som 2*sqrt(diagonal(var)) ??? ###########")
print(2*np.sqrt(np.diagonal(var)))






############### Plotting
# Plot the surface.
surf = ax.plot_surface(x,y,z,cmap=plt.cm.viridis,linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()


fig = plt.figure()
ax = fig.gca(projection='3d')
#Reshape
zpredplot = zpred.reshape(20,20)
surf2 = ax.plot_surface(x,y,zpredplot,cmap=plt.cm.viridis,linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
plt.show()