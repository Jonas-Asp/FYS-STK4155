# Importing various packages
from random import random, seed
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score


x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
z = (5*x*x )*(5*y)

X = np.c_[np.ones((len(z),1)),x,y
           ,x**2,x*y,y**2
           ,x**3,x**2*y,x*y**2,y**3
           ,x**4,x**3*y,x**2*y**2,x*y**3,y**4
           ,x**5,x**4*y,x**3*y**2,x**2*y**3,x*y**4,y**5]

beta = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(z)

zpred = beta[0]+beta[1]*x+beta[2]*y
zpred = zpred+beta[3]*x**2+beta[4]*x*y+beta[5]*y**2
zpred = zpred+beta[6]*x**3+beta[7]*x**2*y+beta[8]*x*y**2+beta[9]*x*y**3
zpred = zpred+(beta[10]*x**4)+(beta[11]*x**3*y)+(beta[12]*x**2*y**2)+(beta[13]*x*y**3)+(beta[14]*y**4)
zpred = zpred+(beta[15]*x**5)+(beta[16]*x**4*y)+(beta[17]*x**3*y**2)+(beta[18]*x**2*y**3)+(beta[19]*x*y**4)+(beta[20]*y**5)


print(np.diag(np.linalg.inv(X.transpose().dot(X))))

plt.plot(x,z,'ro')
plt.plot(x,zpred,'bo')
plt.show()