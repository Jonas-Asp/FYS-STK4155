import numpy as np
import matplotlib.pyplot as plt

# This function calculates the energies of the states in the nn Ising Hamiltonian
def ising_energies(states,L):
    J=np.zeros((L,L))
    for i in range(L):
        J[i,(i+1)%L]-=1.0
    print(np.shape(J)," J")
    # compute energies
    E = np.einsum('...i,ij,...j->...',states,J,states)
    return E

#----------------------------------------------------------------------------------------------------#
L=40
number_states = int(1e4)
# create 10000 random Ising states
states1=np.random.choice([-1, 1], size=(number_states,L))
print(np.shape(states1), "States1 before")
# calculate Ising energies
energies=ising_energies(states1,L)

# reshape Ising states into RL samples: S_iS_j --> X_p
states=np.einsum('...i,...j->...ij', states1, states1)
shape=states.shape
states=states.reshape((shape[0],shape[1]*shape[2]))
# build final data set
Data=[states,energies]

print(np.shape(Data[0]),"States")
print(np.shape(Data[1]),"Energy")

n_samples=400
# define train and test data sets
X_train=states[:n_samples]
Y_train=Data[1][:n_samples] #+ np.random.normal(0,4.0,size=X_train.shape[0])
X_test=states[n_samples:3*n_samples//2]
Y_test=Data[1][n_samples:3*n_samples//2] #+ np.random.normal(0,4.0,size=X_test.shape[0])
#----------------------------------------------------------------------------------------------------#
Y_train = Y_train.reshape(len(Y_train),1)
Y_test = Y_test.reshape(len(Y_test),1)
#print(np.size(X_train,1))
#print(np.size(Y_train))
#print(np.shape(X_train))
#print(np.shape(Y_train))

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoidPrime(z):
    return np.exp(-z)/(1+(np.exp(-z))**2)

class NeuralNetwork (object):
    def __init__(
            self,
            x,
            y,
            layers,
        ):
        
        self.n_layers = len(layers)
        
        """ initiate weights and biases. 0 is input layer, -1 is output layer"""
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]
        self.weights = [np.random.randn(y, x) for y, x in zip(layers[:-1], layers[1:])]
    
    
    def forward(self, x):
        current_activation = x
        self.activations = [x] # Store activations       
        self.zs = []          # z-store variable
        for w,b in zip(self.weights,self.biases):           
            z = np.dot(current_activation[0], w) + b
            current_activation = sigmoid(z)
            self.zs.append(z)
            self.activations.append(current_activation[0].reshape(-1,1))
            
        
    def backprop(self, x, y):        
        self.forward(x)
        
        a = self.activations
        w = self.weights
        zs = self.zs
        gradients_b = [np.zeros(b.shape) for b in self.biases]
        gradients_w = [np.zeros(w.shape) for w in self.weights]
        
        
        delta = self.cost_derivative(a[-1],y) * sigmoidPrime(zs[-1])
        gradients_b[-1] = delta
        gradients_w[-1] = np.dot(delta, a[-2].transpose())
        
        for i in range(2,self.n_layers): 
            z = zs[-i]
            sp = sigmoidPrime(z)
            delta = np.dot(w[-i+1].transpose(), delta) * sp
               
            gradients_w[-i] = np.dot(a[-i-1].transpose(), delta)
            gradients_b[-i] = delta
            
            
        self.gradients_b = gradients_b 
        self.gradients_w = gradients_w
        
        self.update()
    
    def update(self):
        g_b = self.gradients_b
        g_w = self.gradients_w
        
        #print(np.shape(g_b[0]), np.shape(self.biases[0]))
        
        eta = 1e-3
        
        for i in range(self.n_layers-1):
            self.weights[i] -= eta * g_w[i]
            self.biases[i]  -= eta * g_b[i]
            
    def train(self, x, y):
        epochs = int(1e3)
        mse = []
        for i in range(epochs):
            self.forward(x)
            mse.append(np.square(self.activations[-1] - y))
            self.backprop(x,y)
        return mse
        
    def cost_derivative(self, output_activation, y):
        return (output_activation - y)

       
        
    
 
################################### Input, hidden, output
NN = NeuralNetwork(X_train,Y_train,[1600,400,400,1])
mse = NN.train(X_train, Y_train)