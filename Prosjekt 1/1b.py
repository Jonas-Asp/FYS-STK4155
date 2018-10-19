<<<<<<< HEAD
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 10:33:32 2018
@author: Jonas Asperud
"""
=======
>>>>>>> b623a6cc6ce5c1be6d2aafa502299bfa504467eb
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
from scipy import linalg
from sklearn.metrics import mean_squared_error

<<<<<<< HEAD
 # Bootstrap
def bootstrap(sampleData,nBoots,designMatrix,lam,shape):
    # Initiate matrices
    bootVec = np.zeros(len(sampleData))
    b = np.zeros((nBoots,designMatrix.shape[1]))
    mse = np.zeros(nBoots)
    r2 = np.zeros(nBoots)
    mean = np.zeros(nBoots)
    # Assigns the last third as test data and the first 3 thirds as training data
    threeThirds = int(3*len(sampleData)/4)
    trainingData = sampleData[0:threeThirds]
    testData = sampleData[threeThirds+1:]
    # Loops through the set number of boots
    for k in range(0,nBoots):
        # Chooses a random set of data points from data, with same number of rows as data
        bootVec = np.random.choice(trainingData, len(sampleData))
        # Calculates beta values

        b[k,:] = linalg.inv(designMatrix.T.dot(designMatrix) + lam*np.identity(shape)).dot(designMatrix.T).dot(bootVec)
        # Create a fit model for the data
        bootpred = designMatrix.dot(b[k,:]).flatten()
        # Does error calculation
        mse[k],r2[k] = error(testData,bootpred[threeThirds+1:])
        
        mean[k] = np.average(bootVec)
    
    
    std = np.std(mean)
    var = np.var(mean)
    avg = np.average(mean)

    return np.mean(mse),np.mean(r2),b,std,var,avg

# Creating the FrankeFunction
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

#Bare for ha en enkel plotte funksjon
=======
# Bare for ha en enkel plotte funksjon
>>>>>>> b623a6cc6ce5c1be6d2aafa502299bfa504467eb
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
    ax.view_init(30, 60)
    plt.show()   

<<<<<<< HEAD
# Calculates beta values and predict the function using this fit/model
def regression(z,designMatrix,lam,shape):
    beta = np.zeros((len(lam),shape))
    for k in range(len(lam)):
        beta[k] = linalg.inv(designMatrix.T.dot(designMatrix)+lam[k]*np.identity(shape)).dot(designMatrix.T).dot(z)
    return beta
=======
# Creating the FrankeFunction
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4
>>>>>>> b623a6cc6ce5c1be6d2aafa502299bfa504467eb

# Do the regression and return beta values
def regression(z,designMatrix,lam):
    return linalg.inv(designMatrix.T.dot(designMatrix)+lam*np.identity(designMatrix.shape[1])).dot(designMatrix.T).dot(z)

# Create design matrix
def designMatrix(X,Y,degree):
    XY = np.c_[X,Y]
    degree = PolynomialFeatures(degree=degree)
    return degree.fit_transform(XY)

def MSE(z,zpred):
    return 1/len(z) * sum((z-zpred)**2)

def variance(zpred):
    return np.mean(np.mean(zpred**2, axis=1)-np.mean(zpred, axis=1)**2)

def Bias(z,zpred):
    return np.mean((z-np.mean(zpred, axis=1))**2)

<<<<<<< HEAD
################### Setting data ##########################
=======
def bootstrap(x,y,z,degree,nboots,lam):
    # Split into training and test data 3/4 to 1/4 ratio
    xtrain,xtest,ytrain,ytest,ztrain,ztest = train_test_split(x,y,z, test_size=0.25)
    
    mse = np.zeros(nboots)
    zpred = np.zeros((ztest.shape[0],nboots))
    for i in range(nboots-1):
        xboot,yboot,zboot = resample(xtrain,ytrain,ztrain)
        
        dmtrain = designMatrix(xboot,yboot,degree)
        
        beta = regression(zboot,dmtrain,lam)
        
        dmtest = designMatrix(xtest,ytest,degree)
        zpred[:,i] = dmtest.dot(beta)
        
        mse[i] = MSE(ztest,zpred[:,i])
    
    var = variance(zpred)
    bias = Bias(ztest,zpred)
    std = np.std(mse)
    return np.mean(mse),std,bias, np.mean(var)

>>>>>>> b623a6cc6ce5c1be6d2aafa502299bfa504467eb
# Make data.
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)
z = FrankeFunction(x,y)

<<<<<<< HEAD
# Setting the right format for the matrices and add noise
Z = z.flatten()+0*np.random.rand(z.shape[0]*z.shape[1],1).flatten()
=======
>>>>>>> b623a6cc6ce5c1be6d2aafa502299bfa504467eb
X = x.flatten()
Y = y.flatten()
Z = z.flatten() + 0*np.random.rand(z.shape[0]*z.shape[1],1).flatten()

degree = 5
lam = np.linspace(0,1,10)
mse = np.zeros(len(lam))
bootmse = np.zeros(len(lam))
bootstd = np.zeros(len(lam))
bootbias = np.zeros(len(lam))
bootvar = np.zeros(len(lam))
for i in range(len(lam)):
    dm = designMatrix(X,Y,degree)
    beta = regression(Z,dm,lam[i])
    
    zpred = dm.dot(beta)
    #plotting(x,y,zpred.reshape(20,20))
    mse[i] = MSE(Z,zpred)
    bootmse[i],bootstd[i],bootbias[i],bootvar[i] = bootstrap(X,Y,Z,degree,100,lam[i])

#### Plot bootrap mot real !? Uten noise
plt.plot(lam,mse,'o-',label='Ekte bruttovarians')
plt.errorbar(lam, bootmse,2*bootstd, ls='--', marker='o',capsize=5,label='Bootrap bruttovarians')
plt.legend()
plt.xlabel('Lambda verdi')
plt.ylabel('Bruttovarians')
plt.title('Bruttovarians forskjell mellom ekte funksjon og bootstrap med 95% konfidensintervall ')
plt.xticks(lam)
plt.show()

### Varians bias decomposition
plt.plot(lam,bootvar,'o-',label='Varians')
plt.plot(lam,bootbias,'o-',label='Forventningsrett')
plt.plot(lam,bootmse,'--',label='Bootstrap bruttovarians')
plt.xticks(lam)
plt.legend()
<<<<<<< HEAD
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
=======
plt.xlabel('Lambda verdi')
plt.ylabel('Arb. Unit')
plt.title('Forventningsrett-Varians dekomposisjon')
plt.show()
>>>>>>> b623a6cc6ce5c1be6d2aafa502299bfa504467eb
