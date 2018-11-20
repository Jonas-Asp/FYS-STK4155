from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import pandas as pd 
import pickle
import numpy as np

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
    


pickle_out = open("df.pickle","wb")
pickle.dump(df_matrix, pickle_out)
pickle_out.close()

print("Done")