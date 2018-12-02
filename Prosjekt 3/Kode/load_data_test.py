import pickle
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns

""" Load the python object from the excel datafile """
pickle_in = open("df.pickle","rb")
df = pickle.load(pickle_in)
pickle_in.close()




""" Testing data """
# import datafile
file = r'data/default_of_credit_card_clients.xls'
# read datafile
df = pd.read_excel(file)

print(df.head())
df.info() # output shown below

corr = df.corr()
print(corr)
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)
plt.title("Korrelasjons matrisen til datsettet", fontsize=18,fontweight="bold", y=1.04)
plt.show()

outcomes = np.arange(2)
default_sum = np.sum(df.default_payment_next_month)
no_default = len(df.default_payment_next_month) - default_sum
plt.bar(outcomes, [default_sum, no_default], color=['green', 'orange'])
plt.xticks(np.arange(2), ('Klarer ikke å betale tilbake', 'Klarer å betale tilbake'),fontsize=12)
plt.title("Tibakebetalings antall - Oktober 2005", fontsize=18, fontweight="bold", y=1.04)
plt.ylabel("Antall individer",fontsize=14)
plt.show()