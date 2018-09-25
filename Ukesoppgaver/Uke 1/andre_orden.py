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

x = np.random.rand(100,1)
y = (5*x*x )*0.1*np.random.rand(100,1)

xb = np.c_[np.ones((100,1)), x,x**2,x**3,x**4,x**5]
poly = PolynomialFeatures()

beta = np.linalg.inv(xb.T.dot(xb)).dot(xb.T).dot(y)

# The beta values as polynomial
print('The beta values give the model: %.2f + %.2fx + %.2fx^2 + %.2fx^3 + %.2fx^4 + %.2fx^5' %(beta[0],beta[1],beta[2],beta[3],beta[4],beta[5]))
ypredict = beta[0]+(beta[1]*x)+(beta[2]*x**2)

xnew = np.linspace(0,1,100)
ypred = beta[0]+(beta[1]*xnew)+(beta[2]*xnew**2)+(beta[3]*xnew**3)+(beta[4]*xnew**4)+(beta[5]*xnew**5)

sigma = mean_squared_error(y,ypred)*(100/(100-5-1))
var = np.linalg.inv(xb.T.dot(xb)).dot(sigma)
print(np.diagonal(var))

print("MSE %.2f" %mean_squared_error(y,ypred))
print("R2 %.2f" %r2_score(y,ypred))


avg = bootstrap(ypred.flatten())
print(avg)
print(np.mean(y),np.var(y))

plt.plot(x,y,'ro')
plt.plot(xnew,ypred)
plt.show()