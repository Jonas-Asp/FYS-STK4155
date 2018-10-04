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
    ax.view_init(30, 60)
    plt.show()   

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

def bootstrap(x,y,z,degree,nboots):
    # Split into training and test date 3/4 to 1/4 ratio
    xtrain,xtest,ytrain,ytest,ztrain,ztest = train_test_split(x,y,z, test_size=0.25)
    
    mse = np.zeros(nboots)
    zpred = np.zeros((ztest.shape[0],nboots))
    for i in range(nboots-1):
        xboot,yboot,zboot = resample(xtrain,ytrain,ztrain)
        
        dmtrain = designMatrix(xboot,yboot,degree)
        
        beta = regression(zboot,dmtrain)
        
        dmtest = designMatrix(xtest,ytest,degree)
        zpred[:,i] = dmtest.dot(beta)
        
        mse[i] = MSE(ztest,zpred[:,i])
    
    var = variance(zpred)
    bias = Bias(ztest,zpred)
    std = np.std(mse)
    return np.mean(mse),std,bias, np.mean(var)

# Make data.
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)
z = FrankeFunction(x,y)

X = x.flatten()
Y = y.flatten()
Z = z.flatten() + 0*np.random.rand(z.shape[0]*z.shape[1],1).flatten()

degree = 10
mse = np.zeros(degree)
bootmse = np.zeros(degree)
bootstd = np.zeros(degree)
bootbias = np.zeros(degree)
bootvar = np.zeros(degree)
for i in range(degree):
    dm = designMatrix(X,Y,i+1)
    beta = regression(Z,dm)
    
    zpred = dm.dot(beta)
    #plotting(x,y,zpred.reshape(20,20))
    mse[i] = MSE(Z,zpred)
    bootmse[i],bootstd[i],bootbias[i],bootvar[i] = bootstrap(X,Y,Z,i+1,100)


#### Plot bootrap mot real !? Uten noise
plt.plot(np.linspace(1,degree,degree),mse,'o-',label='Ekte bruttovarians')
plt.errorbar(np.linspace(1,degree,degree), bootmse,2*bootstd, ls='--', marker='o',capsize=5,label='Bootrap bruttovarians')
plt.xticks(np.linspace(1,degree,degree))
plt.legend()
plt.xlabel('Polynomgrad')
plt.ylabel('Bruttovarians')
plt.title('Bruttovarians forskjell mellom ekte funksjon og bootstrap med 95% konfidensintervall ')
plt.show()


### Varians bias decomposition
bootvar = bootvar - bootvar[0]
plt.plot(np.linspace(1,degree,degree),bootvar,'o-',label='Varians')
plt.plot(np.linspace(1,degree,degree),bootbias,'o-',label='Forventningsrett')
plt.plot(np.linspace(1,degree,degree),bootmse,'--',label='Bootstrap bruttovarians')
plt.xticks(np.linspace(1,degree,degree))
plt.legend()
plt.title('Forventningsrett-Varians dekomposisjon')
plt.show()