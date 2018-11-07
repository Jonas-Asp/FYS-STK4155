import numpy as np
import matplotlib.pyplot as plt

# This function calculates the energies of the states in the nn Ising Hamiltonian
def ising_energies(states,L):
    J=np.zeros((L,L))
    for i in range(L):
        J[i,(i+1)%L]-=1.0
    print(J,"Find J")
    # compute energies
    E = np.einsum('...i,ij,...j->...',states,J,states)
    return E

#----------------------------------------------------------------------------------------------------#
L=2

# create 10000 random Ising states
states=np.random.choice([-1, 1], size=(6,L))
print(states)
# calculate Ising energies
energies=ising_energies(states,L)

# reshape Ising states into RL samples: S_iS_j --> X_p
states=np.einsum('...i,...j->...ij', states, states)
shape=states.shape
states=states.reshape((shape[0],shape[1]*shape[2]))
# build final data set
Data=[states,energies]

print(Data[0],"States")
print(Data[1],"Energy")

n_samples=400
# define train and test data sets
X_train=Data[0][:n_samples]
Y_train=Data[1][:n_samples] #+ np.random.normal(0,4.0,size=X_train.shape[0])
X_test=Data[0][n_samples:3*n_samples//2]
Y_test=Data[1][n_samples:3*n_samples//2] #+ np.random.normal(0,4.0,size=X_test.shape[0])
#----------------------------------------------------------------------------------------------------#
x = X_train # 4 x 4
y = Y_train.reshape(x.shape[0],1) # 4 x 1

class NeuralNetwork:
    def __init__(
        self,
        X_data,
        Y_data,
        n_hidden_neurons=50,
        n_categories=1,
        n_outputs = 6,
        eta=0.1,
        lmbd=0.0,

    ):
        self.X_data = X_data
        self.Y_data = Y_data

        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.n_hidden_neurons = n_hidden_neurons
        self.n_categories = n_categories
        self.n_outputs = n_outputs
        
        self.eta = eta
        
        self.create_biases_and_weights()

    def create_biases_and_weights(self):
        self.hidden_weights = np.random.randn(self.n_features, self.n_hidden_neurons)
        self.hidden_bias = np.zeros(self.n_hidden_neurons) + 0.01

        self.output_weights = np.random.randn(self.n_outputs, self.n_categories)
        self.output_bias = np.zeros(self.n_categories) + 0.01

    def feed_forward(self):
        # feed-forward for training
        self.z_h = np.matmul(self.X_data, self.hidden_weights) + self.hidden_bias
        self.a_h = self.sigmoid(self.z_h)

        self.z_o = np.matmul(self.a_h, self.output_weights) + self.output_bias
        self.a_o = self.sigmoid(self.z_o)


    def backpropagation(self):
        delta_o = (self.z_o - self.Y_data) * self.sigmoidPrime(self.z_o)
        grad_o_w = np.dot(self.a_o.T,delta_o)
        grad_o_b = delta_o
        
        
        delta_h = np.dot(delta_o,self.output_weights.T) * self.sigmoidPrime(self.z_h)
        grad_h_w = np.dot(self.X_data.T,delta_h)
        grad_h_o = np.sum(delta_h,axis=0)

        """ Update weights and biases """
        print(np.shape(self.output_weights))
        print(np.shape(grad_o_w))
        self.output_weights -= self.eta * grad_o_w
        self.output_bias -= self.eta * grad_o_b
        
    def sigmoid(self, z):
        # Apply sigmoid activation function
        return 1./(1.+np.exp(-z))
    
    def sigmoidPrime(self,z):
    #Derivative of the sigmoid function
        return self.sigmoid(z)*(1-self.sigmoid(z))

NN = NeuralNetwork(x,y)
NN.feed_forward()
NN.backpropagation()
        