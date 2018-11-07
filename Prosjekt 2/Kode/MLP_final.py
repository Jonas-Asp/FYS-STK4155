

# X = (hours sleeping, hours studying), y = Score on test
X = np.array(([3,5], [5,1], [10,2]), dtype=float)
y = np.array(([75], [82], [93]), dtype=float)

x = X/np.amax(X, axis=0)
y = y/100 #Max test score is 100

print(np.shape(x))

class NeuralNetwork:
    def __init__(
        self,
        x,
        y,
        n_input  = np.size(x,1),
        n_hidden = 10,
        n_output = 1,
        eta = 1e-2
        

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
for i in range(int(1e4)):
    NN.forward()
    NN.backpropagation()
    mse.append(sum(np.square(NN.a_o - y)))


plt.plot(np.linspace(1,len(mse),len(mse)),mse, label=min(mse))
plt.legend()
plt.show()
print(NN.a_o.T)
print("------------------")
print(y.T)
#print("-------------------")
#print(NN.o_weights.T)




