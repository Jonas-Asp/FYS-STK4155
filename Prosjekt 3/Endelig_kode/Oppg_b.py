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

def cumulative(z,ytest):
    s = skplt.metrics.plot_cumulative_gain(ytest, z)
    plt.show()
    zsum = sum(z[:,1])
    tmax = 15000
        
    a = s.axes.get_children()
    gain = a[1].get_data()
    baseline = a[2].get_data()
    
    bestx = np.linspace(0,tmax,tmax+1)
    besty = np.concatenate((np.linspace(0,zsum,zsum+1) ,np.linspace(zsum,zsum,tmax-zsum+1)),axis=0)
    
    plt.plot(gain[0][:]*tmax,gain[1][:]*zsum, label="Modell kurve")
    plt.plot(baseline[0][:]*tmax,baseline[1][:]*zsum, '--', label="Baseline")
    plt.plot(bestx,besty, label="Teoretisk beste kurve")
    plt.title("Gains kurven ved logistisk regresjon", fontsize = 16, fontweight="bold", y=1.08)
    plt.ylabel("Kumulativ target data",fontsize=14)
    plt.xlabel("Test sett størrelse",fontsize=14)
    plt.legend()
    plt.plot(x,y)
    plt.show()
    
    gmb = sum(gain[1][:]*zsum - np.linspace(0,zsum,tmax+1))
    bmg = sum(besty[:] - gain[1][:]*zsum)
    print("Area ratio %.10f" %(gmb/bmg))

def bootstrap(xtrain,xtest,ytrain,ytest,nboots):
    default = np.zeros((ytrain.shape[0],nboots))
    nodefualt = np.zeros((ytrain.shape[0],nboots))
    accuarcy = np.zeros(nboots)

    for i in range(nboots):
        xboot,yboot = resample(xtrain,ytrain)
        logreg = LogisticRegression().fit(xboot,yboot)
        hold = logreg.predict_proba(xtest)

        accuarcy[i] = logreg.score(xtest,ytest)
        nodefualt[:,i] = hold[:,0]
        default[:,i] = hold[:,1]
        
    nodefa = np.mean(nodefualt, axis=1)
    defa = np.mean(default, axis=1)
    ypred = np.c_[(nodefa,defa)]
    return ypred, np.mean(accuarcy)



""" Load the python object from the excel datafile """
pickle_in = open("df.pickle","rb")
df = pickle.load(pickle_in)
pickle_in.close()

y = df[:,23]
x = df[:,:23]


""" Uten bootstrap """
# Split 50-50
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

logreg = LogisticRegression().fit(x_train,y_train)
z = logreg.predict_proba(x_test)



cumulative(z,y_test)
# Accuracy score
accuracy = logreg.score(x_test,y_test)
print("accuracy = ", accuracy * 100, "%")
print("Sannsynlig antall som ikke klarer å betaler, %.10f" %sum(z[:,1]))
print("Sannsynlig antall som klarer å betaler, %.10f" %sum(z[:,0]))

coeff = list(logreg.coef_[0])
labels = list(x_train)
features = pd.DataFrame()
features['Features'] = ['LIMIT_BAL',	'SEX',	'EDUCATION',	'MARRIAGE',	'AGE'	,'PAY_0',	'PAY_2',	'PAY_3',	'PAY_4',	'PAY_5',	'PAY_6',	'BILL_AMT1',	'BILL_AMT2',	'BILL_AMT3',	'BILL_AMT4',	'BILL_AMT5',	'BILL_AMT6',	'PAY_AMT1',	'PAY_AMT2',	'PAY_AMT3',	'PAY_AMT4',	'PAY_AMT5',	'PAY_AMT6']
features['importance'] = coeff
features.sort_values(by=['importance'], ascending=True, inplace=True)
features['positive'] = features['importance'] > 0
features.set_index('Features', inplace=True)
features.importance.plot(kind='barh', figsize=(11, 6),color = features.positive.map({True: 'blue', False: 'red'}))
plt.xlabel('Viktighet',fontsize=14)
plt.ylabel('Variabler',fontsize=14)
plt.title("Bidraget fra koeffisentene - Logistisk regresjon m/bootstrap", fontsize = 16, fontweight="bold", y=1.02)
plt.show()


""" Med bootstrap """

nboots = 50
z2,ac = bootstrap(x_train,x_test,y_train,y_test,nboots)
cumulative(z2,y_test)
print("accuracy = ", ac * 100, "%")
print("Sannsynlig antall som ikke klarer å betaler, %.10f" %sum(z2[:,1]))
print("Sannsynlig antall som klarer å betaler, %.10f" %sum(z2[:,0]))


coeff = list(logreg.coef_[0])
labels = list(x_train)
features = pd.DataFrame()
features['Features'] = ['LIMIT_BAL',	'SEX',	'EDUCATION',	'MARRIAGE',	'AGE'	,'PAY_0',	'PAY_2',	'PAY_3',	'PAY_4',	'PAY_5',	'PAY_6',	'BILL_AMT1',	'BILL_AMT2',	'BILL_AMT3',	'BILL_AMT4',	'BILL_AMT5',	'BILL_AMT6',	'PAY_AMT1',	'PAY_AMT2',	'PAY_AMT3',	'PAY_AMT4',	'PAY_AMT5',	'PAY_AMT6']
features['importance'] = coeff
features.sort_values(by=['importance'], ascending=True, inplace=True)
features['positive'] = features['importance'] > 0
features.set_index('Features', inplace=True)
features.importance.plot(kind='barh', figsize=(11, 6),color = features.positive.map({True: 'blue', False: 'red'}))
plt.xlabel('Viktighet',fontsize=14)
plt.ylabel('Variabler',fontsize=14)
plt.title("Bidraget fra koeffisentene - Logistisk regresjon m/bootstrap", fontsize = 16, fontweight="bold", y=1.02)
plt.show()