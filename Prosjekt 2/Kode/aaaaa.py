# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 14:49:05 2018

@author: Rafiki
"""

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
            x_test,
            y_test,
            n_extra_h,
            n_input    = np.size(X_train,1), # 2
            n_hidden   = np.size(Y_train),  # 3
            n_output   = 1,
            epochs     = int(1e2),
            eta        = 1e-3,
            lmbd       = 1e-0
        ):
        
        self.n_hidden   = n_hidden
        self.n_output   = n_output
        self.n_input    = n_input
        self.n_extra_h  = n_extra_h
        
        self.x      = x
        self.y      = y
        self.x_test = x_test
        self.y_test = y_test
        
        self.eta    = eta
        self.lmbd   = lmbd
        self.epochs = epochs
        
        
        self.initiate_weights_and_bias()
        if(self.n_extra_h[0] != 0):
            self.extra_hidden_layers()
        
        print("-----------------------")
        print("      Epochs  =", self.epochs)
        print("      Eta     =", self.eta)
        print("      Lambda  =", self.lmbd)
        print("Layer Rows    =", self.n_input, self.n_extra_h[1:], self.n_hidden)
        print("Layer Columns =", self.n_hidden, self.n_extra_h[:-1], self.n_output)
        print("-----------------------")
              
    def initiate_weights_and_bias(self):
        self.h_weights = np.random.randn(self.n_input, self.n_hidden)
        self.h_bias    = np.random.randn(self.n_hidden) + 0.001
        
        self.o_weights = np.random.randn(self.n_hidden, self.n_output) 
        self.o_bias    = np.random.randn(self.n_output) + 0.001
        
        
    def extra_hidden_layers(self):
        self.n_extra_h     = np.insert(self.n_extra_h,0,self.n_hidden)
        self.extra_weights = []
        self.extra_bias    = []
        
        for i in range(len(self.n_extra_h)-1):
            self.extra_weights.append(np.random.randn(self.n_extra_h[i], self.n_extra_h[i+1]))
            self.extra_bias.append(np.random.randn(self.n_extra_h[i+1]) + 0.001)
        
        self.o_weights = np.random.randn(self.n_extra_h[-1], self.n_output) 
        self.o_bias    = np.random.randn(self.n_output) + 0.001

        
    def forward(self):
        self.z_h = np.dot(self.x, self.h_weights) + self.h_bias
        self.a_h = sigmoid(self.z_h)
        
        # If there is extra layers
        if(self.n_extra_h[0] != 0):
            self.extra_z = []
            self.extra_a = []
            
            for i in range(len(self.n_extra_h)-1):
                self.extra_z.append(np.dot(self.a_h, self.extra_weights[0]) + self.extra_bias[0])
                self.extra_a.append(sigmoid(self.extra_z[0]))
            
            self.z_o  = np.dot(self.extra_a[-1], self.o_weights) + self.o_bias
            self.yHat = sigmoid(self.z_o)
        else:
            self.z_o  = np.dot(self.a_h, self.o_weights) + self.o_bias
            self.yHat = sigmoid(self.z_o)
    
    def backprop(self):
        
        
        if(self.n_extra_h[0] != 0):
            self.d_extra         = []
            self.d_extra_weights = []
            self.d_extra_bias    = []
            
            self.d_o         = np.multiply(-(self.y-self.yHat), sigmoidPrime(self.z_o))
            self.d_o_weights = np.dot(self.extra_a[0].T, self.d_o)
            self.d_o_bias    = np.sum(self.d_o,axis=0)
            
            self.d_extra.append(np.dot(self.d_o, self.o_weights.T) * sigmoidPrime(self.extra_z[0]))
            self.d_extra_weights.append(np.dot(self.a_h.T, self.d_extra[0]))
            self.d_extra_bias.append(np.sum(self.d_extra[0],axis=0))
            
            self.d_h         = np.dot(self.d_extra[0], self.extra_weights[0].T) * sigmoidPrime(self.z_h)
            self.d_h_weights = np.dot(self.x.T, self.d_h)
            self.d_h_bias    = np.sum(self.d_h,axis=0)
        
        else:
            self.d_o         = np.multiply(-(self.y-self.yHat), sigmoidPrime(self.z_o))
            self.d_o_weights = np.dot(self.a_h.T, self.d_o)
            self.d_o_bias    = np.sum(self.d_o,axis=0)
            
            self.d_h         = np.dot(self.d_o, self.o_weights.T) * sigmoidPrime(self.z_h)
            self.d_h_weights = np.dot(self.x.T, self.d_h)
            self.d_h_bias    = np.sum(self.d_h,axis=0)

        self.update()
        
    def update(self):
        if self.lmbd > 0.0:
            self.d_o_weights += self.lmbd * self.o_weights
            self.d_h_weights += self.lmbd * self.h_weights
        
        self.o_weights -= self.eta * self.d_o_weights
        self.o_bias    -= self.eta * self.d_o_bias
        
        self.h_weights -= self.eta * self.d_h_weights
        self.h_bias    -= self.eta * self.d_h_bias
    
    def train(self):
        mse_train = []
        mse_test = []
        
        for i in range(self.epochs):                
                self.forward()
                mse_train.append(sum(np.square(self.y - self.yHat)))
                #mse_test.append(sum(np.square(self.y_test - self.test_set()))) 
                self.backprop()            
        return mse_train, mse_test
    
    def test_set(self):
        z_h = np.dot(self.x_test, self.h_weights) + self.h_bias # 200 x 400
        a_h = sigmoid(z_h)                                      # 200 x 400
        
        z_o  = np.dot(a_h, self.o_weights) + self.o_bias
        yHat = sigmoid(z_o)
        return yHat

NN = NeuralNetwork(X_train,Y_train,X_test,Y_test,[50])
mse_train,mse_test = NN.train()


plt.plot(np.linspace(1,len(mse_train),len(mse_train)),mse_train, 'o', label="Training")
#plt.plot(np.linspace(1,len(mse_test),len(mse_test)),mse_test, 'g--', label="Test")
plt.legend()
plt.show()

#print(NN.yHat.T)
#print("-------------")
#print(NN.y.T)
#print("-------------")
#print(min(mse_test))



