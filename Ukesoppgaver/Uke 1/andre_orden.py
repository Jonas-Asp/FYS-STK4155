#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 13:43:23 2018

@author: Rafiki
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.random.rand(100,1)
y = 5*x*x+0.1*np.random.randn(100,1)

# Adds a column of ones in the first row
xb = np.c_[np.ones((100,1)), x]
#Finner beta
beta = np.linalg.inv(xb.T.dot(xb)).dot(xb.T).dot(y)

xnew = np.array([[0],[2]])
xbnew = np.c_[np.ones((2,1)), xnew]

ypredict = xbnew.dot(beta)
print(xbnew)

plt.plot(xnew, ypredict, "r-")
plt.plot(x, y ,'ro')
plt.axis([0,2.0,0, 15.0])
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'Linear Regression')
plt.show()
