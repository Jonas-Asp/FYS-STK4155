import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import PolynomialFeatures
from scipy import linalg
import matplotlib.pyplot as plt
from sklearn import linear_model
import sklearn as skl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import mean_squared_error, r2_score

# Do the regression and return beta values
def regression(z,designMatrix,lam):
    #return v.T.dot(linalg.pinv(linalg.diagsvd(s, u.shape[0], v.shape[0])+lam*np.identity(designMatrix.shape[1]))).dot(u.T).dot(Y_train)
    return linalg.inv(designMatrix.T.dot(designMatrix)+lam*np.identity(designMatrix.shape[1])).dot(designMatrix.T).dot(z)

# Create design matrix
def designMatrix(X,degree):
    XY = np.c_[X]
    degree = PolynomialFeatures(degree=degree,include_bias=False)
    return degree.fit_transform(XY)

def MSE(z,zpred):
    return 1/len(z) * sum((z-zpred)**2)

def R2(z,zpred,mse):
    return 1 - ((mse*len(z)) / (sum((z-Mean(zpred))**2)))

def Mean(zpred):
    return sum(zpred)/len(zpred)


# The ising model
def ising_energies(states,L):
    """
    This function calculates the energies of the states in the nn Ising Hamiltonian
    """
    J=np.zeros((L,L))
    for i in range(L):
        J[i,(i+1)%L]-=1.0
    # compute energies
    E = np.einsum('...i,ij,...j->...',states,J,states)
    
    return E

np.random.seed(12)
### define Ising model aprams
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

# define number of samples
n_samples=400
# define train and test data sets
X_train=Data[0][:n_samples]
Y_train=Data[1][:n_samples] #+ np.random.normal(0,4.0,size=X_train.shape[0])
X_test=Data[0][n_samples:3*n_samples//2]
Y_test=Data[1][n_samples:3*n_samples//2] #+ np.random.normal(0,4.0,size=X_test.shape[0])

lam = 1e-1
cmap_args=dict(vmin=-1., vmax=1., cmap='seismic')
fig, axarr = plt.subplots(nrows=1, ncols=3)

dMatrix = designMatrix(X_train,1)
u, s, v = linalg.svd(dMatrix)
OLS = v.T.dot(linalg.pinv(linalg.diagsvd(s, u.shape[0], v.shape[0]))).dot(u.T).dot(Y_train)
axarr[0].imshow(OLS.reshape((L,L)),**cmap_args)

ridge = regression(Y_train,dMatrix,lam)
axarr[1].imshow(ridge.reshape((L,L)),**cmap_args)

lasso = linear_model.Lasso()
coefs_lasso=[]
lasso.set_params(alpha=lam) # set regularisation parameter
lasso.fit(X_train, Y_train) # fit model
coefs_lasso.append(lasso.coef_) # store weights
J_lasso=np.array(lasso.coef_).reshape((L,L))
axarr[2].imshow(J_lasso,**cmap_args)

OLSpred = dMatrix.dot(OLS)
mse = MSE(Y_train,OLSpred)
print(mse)
r2 = R2(Y_train,OLSpred,mse)
print(r2)

leastsq=linear_model.LinearRegression()
leastsq.fit(X_train, Y_train)
pred = leastsq.predict(X_train)
print(mean_squared_error(pred, Y_train))
print(r2_score(pred,Y_train))

print('##############')

ridgepred = dMatrix.dot(ridge)
rimse = MSE(Y_train,ridgepred)
print(rimse)
ri2 = R2(Y_train,ridgepred,mse)
print(ri2)      
      
ri=linear_model.Ridge()
ri.set_params(alpha=lam) # set regularisation parameter
ri.fit(X_train, Y_train) # fit model 
riPred = ri.predict(X_train)
print(mean_squared_error(riPred, Y_train))
print(r2_score(riPred,Y_train))
plt.show()















