# Importing various packages
from random import random, seed
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import PolynomialFeatures

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

x = np.ones((2,1))*0.1
y = np.ones((2,1))*2

xy = np.c_[x,y]

#xb = np.c_[np.ones((100,1)), x,x**2,x**3,x**4,x**5]
poly = PolynomialFeatures(degree=3)
xb = poly.fit_transform(xy)
print(xb)



