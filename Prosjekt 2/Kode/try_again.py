import numpy as np
import matplotlib.pyplot as plt

# This function calculates the energies of the states in the nn Ising Hamiltonian
def ising_energies(states,L):
    J=np.zeros((L,L))
    for i in range(L):
        J[i,(i+1)%L]-=1.0
    print(np.shape(J),"Find J")
    # compute energies
    E = np.einsum('...i,ij,...j->...',states,J,states)
    return E

def sigmoid(z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))

def sigmoidPrime(z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)

#----------------------------------------------------------------------------------------------------#
L=40
number_states = int(1e4)
# create 10000 random Ising states
states=np.random.choice([-1, 1], size=(number_states,L))
print(np.shape(states), "States before")
# calculate Ising energies
energies=ising_energies(states,L)

# reshape Ising states into RL samples: S_iS_j --> X_p
states=np.einsum('...i,...j->...ij', states, states)
shape=states.shape
states=states.reshape((shape[0],shape[1]*shape[2]))
# build final data set
Data=[states,energies]

print(np.shape(Data[0]),"States")
print(np.shape(Data[1]),"Energy")

n_samples=400
# define train and test data sets
X_train=Data[0][:n_samples]
Y_train=Data[1][:n_samples] #+ np.random.normal(0,4.0,size=X_train.shape[0])
X_test=Data[0][n_samples:3*n_samples//2]
Y_test=Data[1][n_samples:3*n_samples//2] #+ np.random.normal(0,4.0,size=X_test.shape[0])
#----------------------------------------------------------------------------------------------------#
x = X_train # 6 x 4
y = Y_train.reshape(np.size(X_train,0),1) #


class NeuralNetwork:
    def __init__(
        self,
        x,
        y,
        n_input  = np.size(x,1),
        n_hidden = number_states,
        n_output = 1,
        eta = 1e-1
        

    ):
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_input = n_input
        
        self.x = x
        self.y = y
        
        self.eta = eta
        
        self.h_weights = np.random.randn(self.n_input, self.n_hidden) 
        self.h_bias =    np.random.randn(self.n_hidden) + 0.001
        
        self.o_weights = np.random.randn(self.n_hidden, self.n_output) 
        self.o_bias =    np.random.randn(self.n_output) + 0.001

    
    def forward(self):
        self.z_h = np.dot(self.x, self.h_weights) + self.h_bias
        self.a_h = sigmoid(self.z_h)
        
        self.z_o = np.dot(self.a_h, self.o_weights) + self.o_bias
        self.a_o = sigmoid(self.z_o)
        
        
    
    def backpropagation(self):
        d_o = np.multiply(-(self.y-self.a_o), sigmoidPrime(self.z_o))
        d_o_weights = np.dot(self.a_h.T, d_o)
        d_o_bias = np.sum(d_o, axis=0)


        d_h = np.dot(d_o, self.o_weights.T)*sigmoidPrime(self.z_h)
        d_h_weights = np.dot(x.T, d_h)
        d_h_bias = np.sum(d_h, axis=0)
        
        self.o_weights -= self.eta * d_o_weights
        self.o_bias    -= self.eta * d_o_bias
        
        self.h_weights -= self.eta * d_h_weights
        self.h_bias    -= self.eta * d_h_bias
    
NN = NeuralNetwork(x,y)
NN.forward()
mse = []
for i in range(int(1e2)):
    NN.forward()
    NN.backpropagation()
    mse.append(sum(np.square(NN.a_o - y)))


plt.plot(np.linspace(1,len(mse),len(mse)),mse, label=min(mse))
plt.legend()
plt.show()
#print("-------------------")
#print(NN.o_weights.T)




