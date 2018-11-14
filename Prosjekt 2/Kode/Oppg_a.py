import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from scipy import linalg
import matplotlib.pyplot as plt
from sklearn import linear_model
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import warnings
#Comment this to turn on warnings
warnings.filterwarnings('ignore')

def bootstrap(xtrain,xtest,ytrain,ytest,lam,nboots,OLS,lasso):
    mse = np.zeros(nboots)
    r2 = np.zeros(nboots)
    bias = np.zeros(nboots)
    ypred = np.zeros((ytest.shape[0],nboots))
    
    if(lasso == True):
        for i in range(nboots):
            xboot,yboot = resample(xtrain,ytrain)
            lasso = linear_model.Lasso()
            lasso.set_params(alpha=lam)
            lasso.fit(xboot, yboot)
            ypred[:,i] = lasso.predict(xtest)            
            mse[i] = MSE(ytest,ypred[:,i])
            r2[i] = R2(ytest,ypred[:,i],mse[i])
    else:
        for i in range(nboots):
            xboot,yboot = resample(xtrain,ytrain)
            
            dmtrain = designMatrix(xboot,1)
            
            beta = regression(yboot,dmtrain,lam,OLS)
            
            dmtest = designMatrix(xtest,1)
            ypred[:,i] = dmtest.dot(beta)
            
            mse[i] = MSE(ytest,ypred[:,i])
            r2[i] = R2(ytest,ypred[:,i],mse[i])
    
    bias = Bias(ytest,ypred)
    var = Variance(ypred)
    return np.mean(r2),np.mean(mse), bias, var

# Do the regression and return beta values
def regression(y,designMatrix,lam,OLS):
    if(OLS == True):
        u, s, v = linalg.svd(designMatrix)
        return v.T.dot(linalg.pinv(linalg.diagsvd(s, u.shape[0], v.shape[0]))).dot(u.T).dot(y)
    else:
        return linalg.inv(designMatrix.T.dot(designMatrix)+lam*np.identity(designMatrix.shape[1])).dot(designMatrix.T).dot(y)

def getLasso(x,y,lam):
    lasso = linear_model.Lasso()
    lasso.set_params(alpha=lam)
    lasso.fit(X_train, Y_train)
    beta = lasso.coef_
    lpred = lasso.predict(x)
    return beta, lpred

# Create design matrix
def designMatrix(X,degree):
    XY = np.c_[X]
    degree = PolynomialFeatures(degree=degree,include_bias=False)
    return degree.fit_transform(XY)

# Mean squared error
def MSE(z,zpred):
    return 1/len(z) * sum((z-zpred)**2)

# R2 score caluclation
def R2(z,zpred,mse):
    return 1 - ((mse*len(z)) / (sum((z-Mean(zpred))**2)))

# Calculate the mean
def Mean(zpred):
    return sum(zpred)/len(zpred)

def Bias(z,zpred):
    return np.mean((z - np.mean(zpred.T, axis=0, keepdims=True))**2)

def Variance(zpred):
    return np.mean(np.var(zpred.T, axis=0))


# This function calculates the energies of the states in the nn Ising Hamiltonian
def ising_energies(states,L):
    J=np.zeros((L,L))
    for i in range(L):
        J[i,(i+1)%L]-=1.0
    # compute energies
    E = np.einsum('...i,ij,...j->...',states,J,states)
    return E

############# define Ising model prameters
# system size
L=40
# create 10000 random Ising states
states=np.random.choice([-1, 1], size=(10000,L))
# calculate Ising energies
energies=ising_energies(states,L)

# reshape Ising states into RL samples: S_iS_j --> X_p
states=np.einsum('...i,...j->...ij', states, states)
shape=states.shape
states=states.reshape((shape[0],shape[1]*shape[2]))
# build final data set
Data=[states,energies]
#########################################


# define number of samples
n_samples=400
# define train and test data sets
X_train=Data[0][:n_samples]
Y_train=Data[1][:n_samples] #+ np.random.normal(0,4.0,size=X_train.shape[0])
X_test=Data[0][n_samples:3*n_samples//2]
Y_test=Data[1][n_samples:3*n_samples//2] #+ np.random.normal(0,4.0,size=X_test.shape[0])
print(3*n_samples//2)
print(np.shape(Data[1]))

# Calculate the design matrix of first degree
dMatrix = designMatrix(X_train,1)
# Setting lambda parameter
lam = np.logspace(-4, 5, 10)
mse = np.zeros((3,len(lam)))
r2 = np.zeros((3,len(lam)))

boot_r2 = np.zeros((3,len(lam)))
boot_mse = np.zeros((3,len(lam)))
bias = np.zeros((3,len(lam)))
var = np.zeros((3,len(lam)))

bootstraps = 10

for i in range(len(lam)):
    #cmap_args=dict(vmin=-1., vmax=1., cmap='seismic')
    #fig, axarr = plt.subplots(nrows=1, ncols=3)
    
    ### Normal regression
    ols = regression(Y_train,dMatrix,0,True)
    ridge = regression(Y_train,dMatrix,lam[i],False)
    lasso,lassopred = getLasso(X_train,Y_train,lam[i])


    ################ PLotting
#    axarr[0].set_title('$\\mathrm{OLS}$',fontsize=16)
#    axarr[0].tick_params(labelsize=16)
#    axarr[1].set_title('$\\mathrm{Ridge},\ \\lambda=%.4f$' %(lam[i]),fontsize=16)
#    axarr[1].tick_params(labelsize=16)
#    axarr[2].set_title('$\\mathrm{LASSO},\ \\lambda=%.4f$' %(lam[i]),fontsize=16)
#    axarr[2].tick_params(labelsize=16)
#    
#    axarr[0].imshow(ols.reshape((L,L)),**cmap_args)
#    axarr[1].imshow(ridge.reshape((L,L)),**cmap_args)
#    im = axarr[2].imshow(lasso.reshape((L,L)),**cmap_args)
#    
#    divider = make_axes_locatable(axarr[2])
#    cax = divider.append_axes("right", size="5%", pad=0.05)
#    cbar=fig.colorbar(im, cax=cax)
#    cbar.ax.set_yticklabels(np.arange(-1.0, 1.0+0.25, 0.25),fontsize=14)
#    cbar.set_label('$J_{i,j}$',labelpad=-40, y=1.12,fontsize=16,rotation=0)
#    fig.subplots_adjust(right=1.5)
#    
#    plt.show()
    
    
    
    ################ Error calculation
    olsPred = dMatrix.dot(ols)
    ridgePred = dMatrix.dot(ridge)
    
    mse[0,i] = MSE(Y_train,olsPred)
    mse[1,i] = MSE(Y_train,ridgePred)
    mse[2,i] = MSE(Y_train,lassopred)
    
    r2[0,i] = R2(Y_train,olsPred,mse[0,i])
    r2[1,i] = R2(Y_train,ridgePred,mse[1,i])
    r2[2,i] = R2(Y_train,lassopred,mse[2,i])
    
    boot_r2[0,i],boot_mse[0,i],bias[0,i],var[0,i] = bootstrap(X_train,X_test,Y_train,Y_test,lam[i],bootstraps, True, False)
    boot_r2[1,i],boot_mse[1,i],bias[1,i],var[1,i] = bootstrap(X_train,X_test,Y_train,Y_test,lam[i],bootstraps, False, False)
    boot_r2[2,i],boot_mse[2,i],bias[2,i],var[2,i] = bootstrap(X_train,X_test,Y_train,Y_test,lam[i],bootstraps, False, True)

    
    

#### MSE Plotting
plt.semilogx(lam, mse[0,:], 'b',label='Train (OLS)')
plt.semilogx(lam, boot_mse[0,:], 'b--',label='Bootstrap (OLS)',linewidth=1)
plt.semilogx(lam, mse[1,:], 'm',label='Train (Ridge)',linewidth=1)
plt.semilogx(lam, boot_mse[1,:], 'm--',label='Bootstrap (Ridge)',linewidth=1)
plt.semilogx(lam, mse[2,:], 'g',label='Train (Lasso)',linewidth=1)
plt.semilogx(lam, boot_mse[2,:], 'g--',label='Bootstrap (Lasso)',linewidth=1)


plt.legend(loc='upper right',fontsize=12)
plt.ylim([-1, mse.max()*3])
plt.xlim([min(lam), max(lam)])
plt.xlabel(r'$\lambda$',fontsize=14)
plt.ylabel('MSE',fontsize=14)
plt.title("MSE av ising modellen for forskjellig regresjon", fontsize = 16, fontweight="bold", y=1.08)
plt.show()
    
##### R2 plotting
plt.semilogx(lam, r2[0,:], 'b',label='Train (OLS)')
plt.semilogx(lam, boot_r2[0,:], 'b--',label='Bootstrap (OLS)',linewidth=1)
plt.semilogx(lam, r2[1,:], 'm',label='Train (Ridge)',linewidth=1)
plt.semilogx(lam, boot_r2[1,:], 'm--',label='Bootstrap (Ridge)',linewidth=1)
plt.semilogx(lam, r2[2,:], 'g',label='Train (Lasso)',linewidth=1)
plt.semilogx(lam, boot_r2[2,:], 'g--',label='Bootstrap (Lasso)',linewidth=1)


plt.legend(loc='upper right',fontsize=12)
plt.ylim([0, r2.max()*2.6])
plt.xlim([min(lam), max(lam)])
plt.xlabel(r'$\lambda$',fontsize=14)
plt.ylabel('R2 score',fontsize=14)
plt.title("R2 score av ising modellen for forskjellig regresjon", fontsize = 16, fontweight="bold", y=1.08)
plt.show()

##### Bias-Variance decomp OLS
plt.semilogx(lam, bias[0,:], 'b:',label='Bias')
plt.semilogx(lam, var[0,:], 'm--',label='Variance')
plt.semilogx(lam, boot_mse[0,:], 'g',label='Bootstrap MSE')


plt.legend(fontsize=12)
plt.xlim([min(lam), max(lam)])
plt.xlabel(r'$\lambda$',fontsize=14)
plt.ylabel('Bias / Variance / MSE',fontsize=14)
plt.title("Bias-Varians dekomposisjon av OLS", fontsize = 16, fontweight="bold", y=1.08)
plt.show()

##### Bias-Variance decomp Ridge
plt.semilogx(lam, bias[1,:], 'b:',label='Bias')
plt.semilogx(lam, var[1,:], 'm--',label='Variance')
plt.semilogx(lam, boot_mse[1,:], 'g',label='Bootstrap MSE')


plt.legend(fontsize=12)
plt.xlim([min(lam), max(lam)])
plt.xlabel(r'$\lambda$',fontsize=14)
plt.ylabel('Bias / Variance / MSE',fontsize=14)
plt.title("Bias-Varians dekomposisjon av Ridge", fontsize = 16, fontweight="bold", y=1.08)
plt.show()

##### Bias-Variance decomp Lasso
plt.semilogx(lam, bias[2,:], 'b:',label='Bias')
plt.semilogx(lam, var[2,:], 'm--',label='Variance')
plt.semilogx(lam, boot_mse[2,:], 'g',label='Bootstrap MSE')


plt.legend(fontsize=12)
plt.xlim([min(lam), max(lam)])
plt.xlabel(r'$\lambda$',fontsize=14)
plt.ylabel('Bias / Variance / MSE',fontsize=14)
plt.title("Bias-Varians dekomposisjon av Lasso", fontsize = 16, fontweight="bold", y=1.08)
plt.show()