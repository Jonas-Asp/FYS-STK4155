import numpy as np
import matplotlib.pyplot as plt

#Training Data:
trainX = np.array(([3,5], [5,1], [10,2], [6,1.5]), dtype=float)
trainY = np.array(([-75], [82], [93], [70]), dtype=float)

#Testing Data:
testX = np.array(([4, 5.5], [4.5,1], [9,2.5], [6, 2]), dtype=float)
testY = np.array(([70], [89], [85], [75]), dtype=float)

#Normalize:
print(np.amax(trainX, axis=0))
trainX = trainX/np.amax(trainX, axis=0)
print(trainX)
trainY = trainY/100 #Max test score is 100

#Normalize by max of training data:
testX = testX/np.amax(trainX, axis=0)
testY = testY/100 #Max test score is 100

print(np.shape(trainY))

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
            n_input    = np.size(trainX,1), # 2
            n_hidden   = np.size(trainY),  # 3
            n_output   = 1,
            epochs     = int(1e3),
            batch_size = 1,
            eta        = 1e-3
            
        ):
        
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_input  = n_input
        
        
        self.x      = x
        self.y      = y
        self.x_test = x_test
        self.y_test = y_test
        
        self.eta        = eta
        self.batch_size = batch_size
        self.iterations = int(self.n_input / self.batch_size)
        self.epochs     = epochs
        
        self.initiate_weights_and_bias()
        
    def initiate_weights_and_bias(self):
        self.h_weights = np.random.randn(self.n_input, self.n_hidden)
        self.h_bias    = np.random.randn(self.n_hidden) + 0.001
        
        self.o_weights = np.random.randn(self.n_hidden, self.n_output) 
        self.o_bias    = np.random.randn(self.n_output) + 0.001

    def forward(self):
        self.z_h = np.dot(self.x, self.h_weights) + self.h_bias
        self.a_h = sigmoid(self.z_h)
        
        self.z_o  = np.dot(self.a_h, self.o_weights) + self.o_weights
        self.yHat = sigmoid(self.z_o)
    
    def backprop(self):
        self.d_o         = np.multiply(-(self.y-self.yHat), sigmoidPrime(self.z_o))
        self.d_o_weights = np.dot(self.a_h.T, self.d_o)
        self.d_o_bias    = np.sum(self.d_o,axis=0)
        
        self.d_h         = np.dot(self.d_o, self.o_weights.T) * sigmoidPrime(self.z_h)
        self.d_h_weights = np.dot(self.x.T, self.d_h)
        self.d_h_bias    = np.sum(self.d_h,axis=0)

        self.update()
        
    def update(self):
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
            mse_test.append(sum(np.square(self.y_test - self.test_set())))
            self.backprop()
            
        return mse_train, mse_test
    
    def test_set(self):
        z_h = np.dot(self.x_test, self.h_weights) + self.h_bias
        a_h = sigmoid(z_h)
        
        z_o  = np.dot(a_h, self.o_weights) + self.o_weights
        yHat = sigmoid(z_o)
        return yHat
        
NN = NeuralNetwork(trainX,trainY,trainX,testY)
mse_train,mse_test = NN.train()


plt.plot(np.linspace(1,len(mse_train),len(mse_train)),mse_train, label="Training")
plt.plot(np.linspace(1,len(mse_test),len(mse_test)),mse_test, 'g--', label="Test")
plt.legend()
plt.show()

print(NN.yHat.T)
print("-------------")
print(NN.y.T)
print("-------------")
print(min(mse_test))
print(NN.z_o.T)



