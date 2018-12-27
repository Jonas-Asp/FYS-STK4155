import matplotlib.pyplot as plt
import pandas as pd 
import pickle
import numpy as np
import seaborn as sns

# import datafile
file = r'data/default_of_credit_card_clients.xls'
# read datafile
df = pd.read_excel(file)

df_matrix = np.zeros((int(3e4),24))

# Remove object dependency and put into array
i = 0
for key,val in df.items():
    print(key)
    for j in range(len(val)):
        df_matrix[j,i] = val[j]
    df_matrix[:,i] = df_matrix[:,i]/np.max(val)
    i += 1
    

# save picle object for other scripts
pickle_out = open("df.pickle","wb")
pickle.dump(df_matrix, pickle_out)
pickle_out.close()

print("Done")



""" Load the python object from the excel datafile """
pickle_in = open("df.pickle","rb")
df = pickle.load(pickle_in)
pickle_in.close()


""" Testing data """
# import datafile
file = r'data/default_of_credit_card_clients.xls'
# read datafile
df = pd.read_excel(file)

# Cheking data file, if any errors
print(df.head())
df.info()

# Printing correlation matrix and visualizing
corr = df.corr()
print(corr)
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)
plt.title("Korrelasjons matrisen til datsettet", fontsize=18,fontweight="bold", y=1.04)
plt.show()

# Check what the outcome was oktober 2005, the outcome to be predicted
outcomes = np.arange(2)
default_sum = np.sum(df.default_payment_next_month)
no_default = len(df.default_payment_next_month) - default_sum
plt.bar(outcomes, [default_sum, no_default], color=['green', 'orange'])
plt.xticks(np.arange(2), ('Klarer ikke å betale tilbake', 'Klarer å betale tilbake'),fontsize=12)
plt.title("Tibakebetalings antall - Oktober 2005", fontsize=18, fontweight="bold", y=1.04)
plt.ylabel("Antall individer",fontsize=14)
plt.show()