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

# Creating the FrankeFunction
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

# Do the regression and return beta values
def regression(z,designMatrix):
    return linalg.inv(designMatrix.T.dot(designMatrix)).dot(designMatrix.T).dot(z)
     
# Lag design matrise
def designMatrix(X,Y,degree):
    XY = np.c_[X,Y]
    degree = PolynomialFeatures(degree=5)
    return degree.fit_transform(XY)

# Bare for ha en enkel plotte funksjon
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
    plt.show()   

# Error evaluation
def error(z,zpred):
    #error = np.mean(np.mean((z - zpred)**2, axis=1, keepdims=True))
    mse = np.mean((1/len(z))*sum((z-zpred)**2))
    bias = np.mean((z - np.mean(zpred, axis=1, keepdims=True))**2)  
    var = np.mean(np.var(zpred, axis=1, keepdims=True))
    return mse,bias,var

# Bootstrapping
def bootstrap(x,y,z,degree,nbootstraps):
    # Sorting out training data (75%) and testing data(25)% 
    Xtrain, Xtest, Ytrain, Ytest, Ztrain, Ztest = train_test_split(x,y,z, test_size=0.25)
    
    #zPred = np.empty((n_boostraps,Ztest.size))
    zPred = np.empty((Ztest.size,nbootstraps))
    for i in range(nbootstraps):
        bootSample = resample(Xtrain.flatten(),Ytrain.flatten(),Ztrain.flatten())
        
        dMatrixTrain = designMatrix(bootSample[0],bootSample[1],degree)
        
        beta = regression(bootSample[2],dMatrixTrain)
                
        dMatrixTest = designMatrix(Xtest.flatten(),Ytest.flatten(),degree)
        
        zPred[:,i] = dMatrixTest.dot(beta)
    return zPred,Ztest
    

################### Setting data ##########################
# Make data.
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)
z = FrankeFunction(x,y)

degree = np.linspace(1,5,5)
nbootstraps = 100
mse = np.zeros(len(degree))
bias = np.zeros(len(degree))
var = np.zeros(len(degree))
for i in range(len(degree)):
    # Bootstrapping(x,y,z, polynomial degree, Number of bootraps)
    zpred,Ztest = bootstrap(x,y,z,degree[i],nbootstraps)
    
    # Calculating errors errors
    mse[i],bias[i],var[i] = error(Ztest.reshape(Ztest.size,1),zpred)


plt.plot(degree,bias,'o-',label='bias')
plt.plot(degree,var,'o-',label='Var')
plt.plot(degree,mse,'--',label='MSE')
plt.legend()