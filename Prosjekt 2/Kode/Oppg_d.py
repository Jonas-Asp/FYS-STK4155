import numpy as np
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from scipy import linalg
import matplotlib.pyplot as plt
from sklearn import linear_model
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.model_selection import train_test_split
from sklearn.utils import resample



# This function calculates the energies of the states in the nn Ising Hamiltonian
def ising_energies(states,L):
    J=np.zeros((L,L))
    for i in range(L):
        J[i,(i+1)%L]-=1.0
    # compute energies
    E = np.einsum('...i,ij,...j->...',states,J,states)
    return E


#----------------------------------------------------------------------------------------------------#
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
#----------------------------------------------------------------------------------------------------#

eta_vals = np.logspace(-5, 1, 10)
lmbd_vals = np.logspace(-4, 5, 10)
n_hidden_neurons=50
epochs=10


from sklearn.neural_network import MLPRegressor





# store models for later use
DNN_scikit = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
r2 = np.zeros((len(eta_vals), len(lmbd_vals)))
coefs = []
for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals):
        dnn = MLPRegressor(hidden_layer_sizes=(1,5,10,20), activation='logistic',
                            alpha=lmbd, learning_rate_init=eta, max_iter=epochs, solver='sgd')
        dnn.fit(X_train, Y_train)
        
        #cf = (dnn.coefs_[-1])
        

        
        r2[i][j] = dnn.score(X_test, Y_test)
        
        
#        cmap_args=dict(vmin=-1., vmax=1., cmap='seismic')
#        fig, axarr = plt.subplots(nrows=1, ncols=3)
#        
#        ############### PLotting
#        axarr[0].set_title('$\\mathrm{OLS}$',fontsize=16)
#        axarr[0].tick_params(labelsize=16)
#        axarr[1].set_title('$\\mathrm{OLS}$',fontsize=16)
#        axarr[1].tick_params(labelsize=16)
#        axarr[2].set_title('$\\mathrm{OLS}$',fontsize=16)
#        axarr[2].tick_params(labelsize=16)
#        
#        axarr[0].imshow(cf.reshape((L,L)),**cmap_args)
#        axarr[1].imshow(cf.reshape((L,L)),**cmap_args)
#        im = axarr[2].imshow(cf.reshape((L,L)),**cmap_args)
#        
#        divider = make_axes_locatable(axarr[2])
#        cax = divider.append_axes("right", size="5%", pad=0.05)
#        cbar=fig.colorbar(im, cax=cax)
#        cbar.ax.set_yticklabels(np.arange(-1.0, 1.0+0.25, 0.25),fontsize=14)
#        cbar.set_label('$J_{i,j}$',labelpad=-40, y=1.12,fontsize=16,rotation=0)
#        fig.subplots_adjust(right=1.5)
#        
#        plt.show()
        
        
        
        
        
        
        
#        print("Learning rate  = ", eta)
#        print("Lambda = ", lmbd)
#        print("R2 score on test set: ", dnn.score(X_test, Y_test))
#        print()
        print("Inner =", j)
    print("Iterasjoner = ", i)

for i in range(len(eta_vals)-1):
    if(max(abs(r2[i,:])) > 1):
        print(i)
    else:
        plt.semilogx(lmbd_vals, r2[i,:],label=i)
plt.legend()
plt.show()


