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
warnings.filterwarnings("ignore")

""" Load the python object from the excel datafile """
pickle_in = open("df.pickle","rb")
df = pickle.load(pickle_in)
pickle_in.close()

y = df[:,23]
x = df[:,:23]


# Split 50-50
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)


from sklearn.neural_network import MLPClassifier
# store models for later use

eta_vals = np.logspace(-5, 1, 7)
lmbd_vals = np.logspace(-5, 1, 7)
n_hidden_neurons = [20, 40]
epochs = 10

DNN_scikit = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)

for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals):
        dnn = MLPClassifier(hidden_layer_sizes=(n_hidden_neurons), activation='logistic',
                            alpha=lmbd, learning_rate_init=eta, max_iter=epochs)
        dnn.fit(x_train, y_train)
        
        DNN_scikit[i][j] = dnn
          

import seaborn as sns

sns.set()

test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))

for i in range(len(eta_vals)):
    for j in range(len(lmbd_vals)):
        dnn = DNN_scikit[i][j]
         
        test_pred = dnn.predict(x_test)

        test_accuracy[i][j] = accuracy_score(y_test, test_pred)

        


fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
ax.set_title("Test Accuracy", fontsize=16, fontweight="bold", y=1.08)
ax.set_ylabel("$\eta$", fontsize=14)
ax.set_xlabel("$\lambda$", fontsize=14)
plt.show()