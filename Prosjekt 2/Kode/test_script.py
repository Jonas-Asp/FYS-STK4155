import numpy as np
import matplotlib.pyplot as plt

# This function calculates the energies of the states in the nn Ising Hamiltonian
def ising_energies(states,L):
    J=np.zeros((L,L))
    for i in range(L):
        J[i,(i+1)%L]-=1.0
    E = np.einsum('...i,ij,...j->...',states,J,states)
    return E
#----------------------------------------------------------------------------------------------------#
L=4
number_states = int(8)
#####################
states1=np.random.choice([-1, 1], size=(number_states,L))
energies=ising_energies(states1,L)
states=np.einsum('...i,...j->...ij', states1, states1)
shape=states.shape
states=states.reshape((shape[0],shape[1]*shape[2]))
Data=[states,energies]
n_samples=400
X_train=states[:n_samples]
Y_train=Data[1][:n_samples] 
X_test=states[n_samples:3*n_samples//2]
Y_test=Data[1][n_samples:3*n_samples//2] 
#----------------------------------------------------------------------------------------------------#
Y_train = Y_train.reshape(len(Y_train),1)
Y_test = Y_test.reshape(len(Y_test),1)
# Normalize
Y_train = Y_train/np.max(Y_train)

""" Different activation functions: Sigmoid, Tanh, Softsign """
def sigmoid(z):
    return 1/(1+np.exp(-z))
def sigmoidPrime(z):
    return sigmoid(z)*(1-sigmoid(z))
def tanh(z):
    return np.tanh(z)
def tanhPrime(z):
    return 1 - np.square(tanh(z))
def softsign(z):
    return z/(1+np.abs(z))
def softsignPrime(z):
    return 1/np.square(1+np.abs(z))



class NeuralNetwork(object):
    def __init__(
            self,
            x,
            y,
            h_layers,
            n_inputs     = np.size(X_train,1), 
            n_row_output = np.size(Y_train),
            n_col_output = 1,
            epochs       = int(1e4),
            eta          = 1e-1,
            lmbd         = 1e-2
        ):
        """ Setting variables """
        # Creating matrix with all layer sizes
        self.layers = [n_inputs]
        for i in range(len(h_layers)):
            self.layers.append(h_layers[i])
        self.layers.append(n_row_output)
        self.layers.append(n_col_output)
        
        self.x      = x
        self.y      = y
        
        self.eta    = eta
        self.lmbd   = lmbd
        self.epochs = epochs
        
        self.initiate_weights_and_bias()
        
    def initiate_weights_and_bias(self):
        self.weights = []
        self.biases  = []
        l = self.layers
        
        for i in range(len(l)-1):
            self.weights.append(np.random.randn(l[i], l[i+1]))
            self.biases.append(np.random.rand(l[i+1]) + 1e-3)

        
    def forward(self):
        current_activation = self.x
        activations = [current_activation]
        zs = []
        w  = self.weights
        b  = self.biases
        for i in range(len(w)):                   
            z = np.dot(current_activation, w[i]) + b[i]
            zs.append(z)
            current_activation = self.activationFunction(z)
            activations.append(current_activation)

        self.activations = activations
        self.z = zs
        

    def backprop(self,x,y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        a = self.activations
        w = self.weights
        zs = self.z
        
        # First gradient
        delta       = np.multiply(self.cost_derivative(a[-1],y), self.activationPrime(zs[-1]))
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(a[-2].T, delta)
        
        # Rest of the gradients
        for l in range(2, len(self.layers)):
            z = zs[-l]
            fp = self.activationPrime(z)
            delta = np.dot(delta, w[-l+1].T) * fp
            nabla_b[-l] = np.sum(delta,axis=0)
            nabla_w[-l] = np.dot(a[-l-1].T, delta)
            
        
        
        self.nabla_b = nabla_b
        self.nabla_b[-1] = np.sum(delta)
        self.nabla_w = nabla_w
        
        self.update()
    
    def update(self):
        
        for i in range(len(self.weights)):
            if self.lmbd > 0.0:
                self.nabla_w[i] += self.lmbd * self.weights[i]
            self.weights[i] -= self.eta * self.nabla_w[i]
            self.biases[i]  -= self.eta * self.nabla_b[i]

    def training(self,x,y):
        mse = []
        for i in range(self.epochs):
            self.forward()
            mse.append(np.sum(np.square(y - self.activations[-1])))
            self.backprop(x,y)
        return mse
    
    
    def cost_derivative(self, output_activation, y):
        return (output_activation - y)
    
    def activationFunction(self,z):
        return tanh(z)
    
    def activationPrime(self,z):
        return tanhPrime(z)
NN = NeuralNetwork(X_train,Y_train, (10,7))

mse = NN.training(X_train,Y_train)

print(np.min(mse))

x = np.linspace(1,len(mse),len(mse))
plt.plot(x, mse, 'o')

print("-------------")
print(NN.activations[-1].T)
print("-------------")
print(Y_train.T)















