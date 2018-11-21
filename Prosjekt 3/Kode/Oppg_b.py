import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
import pandas as pd 
import scikitplot as skplt

""" Load the python object from the excel datafile """
pickle_in = open("df.pickle","rb")
df = pickle.load(pickle_in)
pickle_in.close()

y = df[:,23]
x = df[:,:23]


# Split 50-50
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

logreg = LogisticRegression().fit(x_train,y_train)
z = logreg.predict_proba(x_test)


# Default-no default in our sample
outcomes = np.arange(4)
default_sum = [np.sum(z[:,1])]
no_default = np.sum(z[:,0])
test_default = sum(y_test)
plt.bar(outcomes, [default_sum, test_default, no_default, abs(test_default - len(y_test))])
plt.xticks(np.arange(4), ('Default','Testset - default','No default', 'Testset - no default'))
plt.show()



# Accuracy score
accuracy = logreg.score(x_test, y_test)
print("accuracy = ", accuracy * 100, "%")

skplt.metrics.plot_cumulative_gain(y_test, z)
plt.xlabel("Nada")
plt.show()


coeff = list(logreg.coef_[0])
labels = list(x_train)
features = pd.DataFrame()
features['Features'] = ['LIMIT_BAL',	'SEX',	'EDUCATION',	'MARRIAGE',	'AGE'	,'PAY_0',	'PAY_2',	'PAY_3',	'PAY_4',	'PAY_5',	'PAY_6',	'BILL_AMT1',	'BILL_AMT2',	'BILL_AMT3',	'BILL_AMT4',	'BILL_AMT5',	'BILL_AMT6',	'PAY_AMT1',	'PAY_AMT2',	'PAY_AMT3',	'PAY_AMT4',	'PAY_AMT5',	'PAY_AMT6']
features['importance'] = coeff
features.sort_values(by=['importance'], ascending=True, inplace=True)
features['positive'] = features['importance'] > 0
features.set_index('Features', inplace=True)
features.importance.plot(kind='barh', figsize=(11, 6),color = features.positive.map({True: 'blue', False: 'red'}))
plt.xlabel('Importance')
plt.show()
