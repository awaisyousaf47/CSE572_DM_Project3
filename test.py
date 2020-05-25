#!/usr/bin/env python
# coding: utf-8

# In[4]:
import K_means
import DB
import numpy as np
import pandas as pd
import sys
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# In[ ]:

test_file_name = sys.argv[1]
kmeans_labels = K_means.km_clustering(test_file_name)
km = kmeans_labels.reshape(((len(kmeans_labels)),1))
dbscan_labels = DB.dbscan_clustering(test_file_name)
dbscan = dbscan_labels.reshape(((len(dbscan_labels)),1))

for i in range(len(kmeans_labels)):
    km[i] = km[i] + 1
    dbscan[i] = dbscan[i] + 1
final = np.append(km, dbscan, axis = 1)
print('Saved the output labels in a csv file')
result = pd.DataFrame(final, columns=['K-Means', 'DBSCAN'])
result.to_csv('Label_Output.csv')

