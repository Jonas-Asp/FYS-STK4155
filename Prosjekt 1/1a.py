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
import pandas as pd

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

def R2(z,zpred,mse):
    return 1 - ((mse*len(z)) / (sum((z-Mean(zpred))**2)))

def Mean(zpred):
    return sum(zpred)/len(zpred)

def variance(zpred):
    return np.mean(np.mean(zpred**2, axis=1)-np.mean(zpred, axis=1)**2)

def Bias(z,zpred):
    return np.mean((z-np.mean(zpred, axis=1))**2)

def CIbeta(designMatrix,mse,degree,N):
    return 2*np.sqrt(np.diagonal(linalg.inv(designMatrix.T.dot(designMatrix)).dot(mse*N/(N-degree-1))))

def bootstrap(x,y,z,degree,nboots):
    # Split into training and test date 3/4 to 1/4 ratio
    xtrain,xtest,ytrain,ytest,ztrain,ztest = train_test_split(x,y,z, test_size=0.25)
    
    mse = np.zeros(nboots)
    r2 = np.zeros(nboots)
    zpred = np.zeros((ztest.shape[0],nboots))
    for i in range(nboots-1):
        xboot,yboot,zboot = resample(xtrain,ytrain,ztrain)
        
        dmtrain = designMatrix(xboot,yboot,degree)
        
        beta = regression(zboot,dmtrain)
        
        dmtest = designMatrix(xtest,ytest,degree)
        zpred[:,i] = dmtest.dot(beta)
        
        mse[i] = MSE(ztest,zpred[:,i])
        r2[i] = R2(ztest,zpred[:,i],mse[i])
    var = variance(zpred)
    bias = Bias(ztest,zpred)
    std = np.std(mse)
    return np.mean(mse),std,bias, np.mean(var),np.mean(r2)

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
r2 = np.zeros(degree)
bootmse = np.zeros(degree)
bootr2 = np.zeros(degree)
bootstd = np.zeros(degree)
bootbias = np.zeros(degree)
bootvar = np.zeros(degree)
ci = np.zeros((degree,int(3)+sum(range(3,degree+2))))
for i in range(degree):
    dm = designMatrix(X,Y,i+1)
    beta = regression(Z,dm)
    
    zpred = dm.dot(beta)
    #plotting(x,y,zpred.reshape(20,20))
    mse[i] = MSE(Z,zpred)
    r2[i] = R2(Z,zpred,mse[i])
    bootmse[i],bootstd[i],bootbias[i],bootvar[i],bootr2[i] = bootstrap(X,Y,Z,i+1,100)
    
    ci[i,:int(3)+sum(range(3,i+3))] = CIbeta(dm,mse[i],degree,len(Z))

noise = np.linspace(0,10,20)
znoise = np.zeros((len(noise),len(Z)))
noisemse = np.zeros((5,len(noise)))
noiser2 = np.zeros((5,len(noise)))
dm = designMatrix(X,Y,5)
for j in range(5):
    for i in range(len(noise)):
        beta = regression(Z*+noise[i]*np.random.rand(Z.size,1).flatten(),dm)
        znoise[i,:] = dm.dot(beta)
        
        noisemse[j,i] = MSE(Z,znoise[i,:])
        noiser2[j,i] = R2(Z,znoise[i,:],noisemse[j,i])
    
#### Plot bootrap mot real !? Uten noise
plt.rcParams['axes.titlepad'] = 15
plt.plot(np.linspace(1,degree,degree),mse,'o-',label='Orginal bruttovarians')
plt.errorbar(np.linspace(1,degree,degree), bootmse,2*bootstd, ls='--', marker='o',capsize=5,label='Bootrap bruttovarians')
plt.xticks(np.linspace(1,degree,degree))
plt.legend()
plt.xlabel('Polynomgrad')
plt.ylabel('Bruttovarians')
plt.title('Bruttovarians - Orginal funksjon og bootstrap m/95% konfidensintervall',fontweight='bold')
plt.show()


### Varians bias decomposition
bootvar = bootvar - bootvar[0]
plt.plot(np.linspace(1,degree,degree),bootvar,'o-',label='Varians')
plt.plot(np.linspace(1,degree,degree),bootbias,'o-',label='Forventningsrett')
plt.plot(np.linspace(1,degree,degree),bootmse,'--',label='Bootstrap bruttovarians')
plt.xticks(np.linspace(1,degree,degree))
plt.legend()
plt.xlabel('Polynomgrad')
plt.ylabel('Bruttovarians / Varians / Forventningsrett')
plt.title('Forventningsrett-Varians dekomposisjon',fontweight='bold')
plt.show()

### Tabulating data
data = np.c_[r2[:5],bootr2[:5],mse[:5],bootmse[:5],bootstd[:5]]
df = pd.DataFrame(data,index=['1','2','3','4','5'],columns=['R2 score',' B-R2 score','MSE','B-MSE','B-Std'])
print(df)

### Beta varians
for i in range(21):
    plt.plot(np.linspace(1,5,5),ci[:5,i],'o-')
plt.xticks(np.linspace(1,5,5))
plt.xlabel('Polynomgrad')
plt.ylabel('Beta varians')
plt.title('Konfidensintervall til beta verdiene',fontweight='bold')
plt.show()

### Plotting noise values
for i in range(5):
    label = ('P = %.0f' %(i+1))
    plt.plot(noise,noisemse[i,:],'-',label=label)
plt.xlabel('Støy varians')
plt.ylabel('Bruttovarians')
plt.title('Bruttovarians ved økende støy m/opptil polynomgrad 5',fontweight='bold')
plt.legend()
plt.show()

for i in range(5):
    label = ('P = %.0f' %(i+1))
    plt.plot(noise,noiser2[i,:],'-',label=label)
plt.xlabel('Støy varians')
plt.ylabel('R2 score')
plt.title('R2 score ved økende støy m/opptil polynomgrad 5',fontweight='bold')
plt.legend()
plt.show()