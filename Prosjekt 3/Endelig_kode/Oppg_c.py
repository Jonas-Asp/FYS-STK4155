import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import scikitplot as skplt
import warnings
import seaborn as sns
from sklearn.neural_network import MLPClassifier
warnings.filterwarnings("ignore")


""" Beregner gains kurven via skplt. Med modell kruve, baseline og teoretisk beste """
def cumulative(z,ytest):
    s = skplt.metrics.plot_cumulative_gain(ytest, z)
    plt.title("Disregard this. Plot as a consequence of skplt")
    plt.show()
    zsum = sum(z[:,1])
    tmax = 15000
        
    a = s.axes.get_children()
    gain = a[1].get_data()
    baseline = a[2].get_data()
    
    bestx = np.linspace(0,tmax,tmax+1)
    besty = np.concatenate((np.linspace(0,zsum,zsum+1) ,np.linspace(zsum,zsum,tmax-zsum+1)),axis=0)
    
    sns.set_style("white")
    plt.figure(facecolor='white')
    plt.plot(gain[0][:]*tmax,gain[1][:]*zsum, label="Modell kurve")
    plt.plot(baseline[0][:]*tmax,baseline[1][:]*zsum, '--', label="Baseline")
    plt.plot(bestx,besty, label="Teoretisk beste kurve")
    plt.title("Gains kurven ved nevrale netverk", fontsize = 16, fontweight="bold", y=1.08)
    plt.ylabel("Kumulativ target data",fontsize=14)
    plt.xlabel("Test sett størrelse",fontsize=14)
    plt.legend()
    plt.plot(x,y)
    plt.show()
    
    gmb = sum(gain[1][:]*zsum - np.linspace(0,zsum,tmax+1))
    bmg = sum(besty[:] - np.linspace(0,zsum,tmax+1))
    print("Area ratio %.10f" %(gmb/bmg))

""" Load the python object from the excel datafile """
pickle_in = open("df.pickle","rb")
df = pickle.load(pickle_in)
pickle_in.close()

y = df[:,23]
x = df[:,:23]


# Split 50-50
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)


""" Grid 1, Check eta and lambda values """
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

""" Grid 2, Check hidden layers and epochs"""
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


""" Gains curve and prediction """
eta = 1e-1
lmbd = 1e-3
n_hidden_neurons = [57, 23]
epochs = 100

dnn = MLPClassifier(hidden_layer_sizes=(n_hidden_neurons), activation='logistic',
                            alpha=lmbd, learning_rate_init=eta, max_iter=epochs)
dnn.fit(x_train, y_train)
test_pred = dnn.predict(x_test)
test_pred2 = dnn.predict_proba(x_test)

accuarcy = accuracy_score(y_test, test_pred)

cumulative(test_pred2,y_test)
print("accuarcy = ", accuarcy * 100, "%")
print("Sannsynlig antall som ikke klarer å betaler, %.10f" %sum(test_pred2[:,1]))
print("Sannsynlig antall som klarer å betaler, %.10f" %sum(test_pred2[:,0]))


