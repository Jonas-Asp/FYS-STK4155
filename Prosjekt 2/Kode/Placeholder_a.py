import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from scipy import linalg
import matplotlib.pyplot as plt
from sklearn import linear_model
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

def bootstrap(x,y,degree,nboots):
    # Split into training and test date 3/4 to 1/4 ratio
    xtrain,xtest,ytrain,ytest= train_test_split(x,y, test_size=0.25)
    
    mse = np.zeros(nboots)
    r2 = np.zeros(nboots)
    ypred = np.zeros((ytest.shape[0],nboots))
    for i in range(nboots-1):
        xboot,yboot = resample(xtrain,ytrain)
        
        dmtrain = designMatrix(xboot,degree)
        
        beta = regression(yboot,dmtrain,0,True)
        
        dmtest = designMatrix(xtest,degree)
        ypred[:,i] = dmtest.dot(beta)
        
        mse[i] = MSE(ytest,ypred[:,i])
        r2[i] = R2(ytest,ypred[:,i],mse[i])
    
    bias = Bias(ytest,ypred)
    
    return np.mean(mse),np.mean(r2)

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
    return np.mean((z - np.mean(zpred, axis=0, keepdims=True))**2)


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

# Calculate the design matrix of first degree
dMatrix = designMatrix(X_train,1)
# Setting lambda parameter
lam = np.logspace(-4, 5, 10)
mse = np.zeros((3,len(lam)))
r2 = np.zeros((3,len(lam)))
boot_mse = np.zeros((len(lam)))
boot_r2 = np.zeros((len(lam)))

for i in range(len(lam)):
    #cmap_args=dict(vmin=-1., vmax=1., cmap='seismic')
    #fig, axarr = plt.subplots(nrows=1, ncols=3)
    
    ### Normal regression
    ols = regression(Y_train,dMatrix,0,True)
    ridge = regression(Y_train,dMatrix,lam[i],False)
    lasso,lassopred = getLasso(X_train,Y_train,lam[i])

    ### Bootstrapping
    boot_mse[i],boot_r2[i] = bootstrap(Data[0],Data[1],1,10)


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


#### MSE Plotting
plt.semilogx(lam, mse[0,:], 'b',label='Train (OLS)')
plt.semilogx(lam, mse[1,:], 'r',label='Train (Ridge)',linewidth=1)
plt.semilogx(lam, mse[2,:], 'g',label='Train (Lasso)',linewidth=1)

plt.legend(loc='upper left',fontsize=12)
plt.ylim([0, mse.max()*2])
plt.xlim([min(lam), max(lam)])
plt.xlabel(r'$\lambda$',fontsize=16)
plt.ylabel('MSE',fontsize=16)
plt.tick_params(labelsize=16)
plt.show()
    
##### R2 plotting
# MSE Plotting
plt.semilogx(lam, r2[0,:], 'b',label='Train (OLS)')
plt.semilogx(lam, r2[1,:], 'r',label='Train (Ridge)',linewidth=1)
plt.semilogx(lam, r2[2,:], 'g',label='Train (Lasso)',linewidth=1)

plt.legend(loc='lower left',fontsize=12)
plt.ylim([0, r2.max()*1.2])
plt.xlim([min(lam), max(lam)])
plt.xlabel(r'$\lambda$',fontsize=16)
plt.ylabel('R2 Score',fontsize=16)
plt.tick_params(labelsize=16)
plt.show()

