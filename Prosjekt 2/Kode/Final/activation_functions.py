import numpy as np
import matplotlib.pyplot as plt

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

x = np.linspace(-6,6,100)

sig = sigmoid(x)
ta  = tanh(x)
soft= softsign(x)

plt.grid(linewidth=0.3)
plt.plot(x,sig)
plt.show()
plt.grid(linewidth=0.3)
plt.plot(x,ta)
plt.show()
plt.grid(linewidth=0.3)
plt.plot(x,soft)
plt.show()