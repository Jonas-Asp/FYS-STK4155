import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
import pandas as pd 
import scikitplot as skplt
from sklearn.utils import resample
import warnings
import seaborn as sns
from sklearn.neural_network import MLPClassifier
warnings.filterwarnings("ignore")

""" Load the python object from the excel datafile """
pickle_in = open("df.pickle","rb")
df = pickle.load(pickle_in)
pickle_in.close()

y = df[:,23]
x = df[:,:23]


# Split 50-50
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)



# store models for later use

eta = 1e-1
lmbd = 1e-3
n_hidden_neurons = [57, 23, 14, 94, 62, 40]
epochs = [10, 50, 100, 250]

DNN_scikit = np.zeros((len(epochs), len(n_hidden_neurons)), dtype=object)


for i in range(len(epochs)):
    for j in range(len(n_hidden_neurons)):
        print(n_hidden_neurons[0:j+1])
        dnn = MLPClassifier(hidden_layer_sizes=(n_hidden_neurons[0:j+1]), activation='logistic',
                            alpha=lmbd, learning_rate_init=eta, max_iter=epochs[i])
        dnn.fit(x_train, y_train)
        
        DNN_scikit[i][j] = dnn
          



sns.set()

test_accuracy = np.zeros((len(epochs), len(n_hidden_neurons)))

for i in range(len(epochs)):
    for j in range(len(n_hidden_neurons)):
        print(n_hidden_neurons[0:j+1])
        dnn = DNN_scikit[i][j]
        
        test_pred = dnn.predict(x_test)
        
        test_accuracy[i][j] = accuracy_score(y_test, test_pred)
        



fig, ax = plt.subplots(figsize = (6, 4))
sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
ax.set_title("Test Accuracy", fontsize=16, fontweight="bold", y=1.08)
ax.set_ylabel("Epochs", fontsize=14)
ax.set_xlabel("Hidden layers", fontsize=14)
plt.show()