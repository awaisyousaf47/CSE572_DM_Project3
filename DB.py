#!/usr/bin/env python
# coding: utf-8

# In[168]:


import pandas as pd
import math
from numpy import zeros
import numpy as np
import csv
from numpy import cov
import matplotlib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import pearsonr
from sklearn.datasets.samples_generator import make_blobs
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition, datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings("ignore")


# In[169]:
def dbscan_clustering(test_file_name):

    mealDataset = pd.DataFrame()
    temp = []
    for i in range(5):
            meal_df = pd.read_csv("mealAmountData{}.csv".format(i+1), sep = '\t', header = None)
            meal_df_arr = np.array(meal_df)
            for i in range(0, 50):
                temp.append(meal_df_arr[i])
                temp_arr = np.asarray(temp)
                t = pd.DataFrame(temp_arr)
    mealDataset = mealDataset.append(t, ignore_index = True)

    mealDataset = t
    mealDataset = mealDataset.replace("NaN", 0)


    # Ground Truth
    def create_bins(lower_bound, width, quantity):
        bins = []
        for low in range(lower_bound,
                         lower_bound + quantity*width + 1, width):
            bins.append((low, low+width))
        return bins

    bins = create_bins(lower_bound=0,
                       width=20,
                       quantity=5)

    bins

    from collections import Counter
    def find_bin(value, bins):
        for i in range(0, len(bins)):
            if value == 0:
                return 0;
            if bins[i][0] < value <= bins[i][1]:
                    return i+1
        return -1

    binned_weights = []
    for value in mealDataset[0]:
        bin_index = find_bin(value, bins)
        bi = bin_index
    #     print(value, bin_index, bins[bin_index])
        binned_weights.append(bin_index)

    frequencies = Counter(binned_weights)
    # print(frequencies)

    train_binned_weights = []
    for i in range(150):
        train_binned_weights.append(binned_weights[i])

    test_binned_weights = []
    for i in range(150, len(binned_weights)):
        test_binned_weights.append(binned_weights[i])

    test_binned_weights = np.array(test_binned_weights)
    # print(binned_weights)


    # In[170]:


    dataset = pd.DataFrame()
    t = []
    for i in range(5):
            df = pd.read_csv("MealNoMealData/mealData{}.csv".format(i+1), sep = '\t', header = None)
            df_arr = np.array(df)
            for i in range(0, 50):
                t.append(df_arr[i])
                t_arr = np.asarray(t)
                t1 = pd.DataFrame(t_arr)
    #         dataset = dataset.append(df, ignore_index = True)

    dataset = t1
    dataset = dataset[0].str.split(',', expand = True)
    dataset = dataset.dropna(how = 'all')
    dataset = dataset.fillna(0)
    dataset = dataset.replace("NaN", 0)
    dataset = dataset.astype(float)
    dataset_arr = np.array(dataset)


    # rolling mean and rolling standard deviation
    data = dataset.T
    rolling_mean = data.rolling(window=3,min_periods=3).mean()
    rolling_mean = rolling_mean.T
    rolling_std = data.rolling(window=3,min_periods=3).std()
    rolling_std = rolling_std.T

    # fft feature
    cgmFFTValues = abs(np.fft.fft(dataset_arr))
    freq = np.fft.fftfreq(dataset_arr.shape[-1])

    # pickign top 8 peaks of FFT
    FFT=np.array(cgmFFTValues)
    fft_freq=np.array(freq)
    Fourier_peak=list()
    Fourier_frequency=list()
    for i in range(len(FFT)):
        index=np.argsort(FFT)[i][-9:]

        peak=FFT[i][index]
        Fourier_peak.append(peak)
        freq=abs(fft_freq[index])
        freq.sort()
        fr=freq[[0,1,3,5,7]]
        Fourier_frequency.append(fr)

    Fourier_peak=np.array(Fourier_peak)
    Fourier_frequency=np.array(Fourier_frequency)
    Fourier_peak=np.unique(Fourier_peak,axis=1)


    # polyfit regression feature
    polyfit_reg = []
    x = [i for i in range(len(dataset_arr[32]))]

    for i in range(len(dataset_arr)):
        polyfit_reg.append(np.polyfit(x, dataset_arr[i], 3))

    polyfit_reg = np.array(polyfit_reg)

    feature_vector=[]
    for idx, row in dataset.iterrows():
        # cgm velocity
        cgm_velocity = [0]
        for index in range(len(row)-1):
            cgm_velocity += [row[index+1]-row[index]]
        cgm_velocity.sort(reverse=True)
        feature_vector += cgm_velocity[:4]

    feature_vector= np.array(feature_vector)
    feature_vector= np.reshape(feature_vector, (250,4))

    # Extracting feature-4: Interquartile Range
    IQR = []
    for i in range(len(dataset_arr)):
        Q1 = np.percentile(dataset_arr[i], 25, interpolation = 'midpoint')
        Q3 = np.percentile(dataset_arr[i], 75, interpolation = 'midpoint')
        IQR.append(Q3 - Q1)

    IQR = np.array(IQR)

    x=list()
    x.append(IQR)
    x=np.array(x)
    x = x.T

    feature_matrix = np.append(rolling_mean, rolling_std, axis = 1)
    feature_matrix = np.append(feature_matrix, Fourier_frequency, axis = 1)
    feature_matrix = np.append(feature_matrix, Fourier_peak, axis = 1)
    feature_matrix = np.append(feature_matrix, polyfit_reg, axis = 1)
    feature_matrix = np.append(feature_matrix, feature_vector, axis = 1)
    feature_matrix = np.append(feature_matrix, x, axis = 1)

    feature_matrix = np.nan_to_num(feature_matrix)
    # sc = StandardScaler()
    sc = MinMaxScaler()
    X_std = sc.fit_transform(feature_matrix)

    from sklearn.preprocessing import StandardScaler

    feature_matrix = StandardScaler().fit_transform(feature_matrix)
    df_feature_matrix = pd.DataFrame(feature_matrix)
    df_feature_matrix.fillna(0, inplace=True)

    from sklearn import preprocessing
    data_scaled = pd.DataFrame(preprocessing.scale(df_feature_matrix), columns=df_feature_matrix.columns)
    pca = decomposition.PCA(n_components=8)
    pca1 = decomposition.PCA(n_components=30)
    X_std_pca = pca.fit_transform(data_scaled)
    X_std_pca1 = pca1.fit_transform(data_scaled)

    pcaTransformed = pca.transform(feature_matrix)
    pcaTransformed1 = pca1.transform(feature_matrix)

    pcaTransformed = pd.DataFrame(pcaTransformed)
    X = pcaTransformed.head(250)

    # X = shuffle(X)
    X = pd.read_csv("db_features.csv", header = None)

    train_X = X.head(150)
    train_X = np.array(train_X)
    # test_X = X.tail(100)
    test_X = pd.read_csv(test_file_name, sep='\t', header=None)
    # test_X = np.array(test_X)
    # X = np.array(X)

    test_X = test_X[0].str.split(',', expand=True)
    test_X = test_X.dropna(how='all')
    test_X = test_X.fillna(0)
    test_X = test_X.replace("NaN", 0)
    test_X = test_X.replace('', 0, regex=True)
    test_X_np = test_X.values
    test_X_1 = test_X_np.astype(float)
    test_X_arr = np.array(test_X_1)

    data = test_X.T
    rolling_mean = data.rolling(window=3, min_periods=3).mean()
    rolling_mean = rolling_mean.T
    rolling_std = data.rolling(window=3, min_periods=3).std()
    rolling_std = rolling_std.T

    # fft feature
    cgmFFTValues = abs(np.fft.fft(test_X_arr))
    freq = np.fft.fftfreq(test_X_arr.shape[-1])

    # pickign top 8 peaks of FFT
    FFT = np.array(cgmFFTValues)
    fft_freq = np.array(freq)
    Fourier_peak = list()
    Fourier_frequency = list()
    for i in range(len(FFT)):
        index = np.argsort(FFT)[i][-9:]

        peak = FFT[i][index]
        Fourier_peak.append(peak)
        freq = abs(fft_freq[index])
        freq.sort()
        fr = freq[[0, 1, 3, 5, 7]]
        Fourier_frequency.append(fr)

    Fourier_peak = np.array(Fourier_peak)
    Fourier_frequency = np.array(Fourier_frequency)
    Fourier_peak = np.unique(Fourier_peak, axis=1)

    # polyfit regression feature
    polyfit_reg = []
    x = [i for i in range(len(test_X_arr[32]))]

    for i in range(len(test_X_arr)):
        polyfit_reg.append(np.polyfit(x, test_X_arr[i], 3))

    polyfit_reg = np.array(polyfit_reg)

    feature_vector = []
    for idx, row in test_X.iterrows():
        # cgm velocity
        cgm_velocity = [0]
        for index in range(len(row) - 1):
            cgm_velocity += [int(row[index + 1]) - int(row[index])]
        cgm_velocity.sort(reverse=True)
        feature_vector += cgm_velocity[:4]

    feature_vector = np.array(feature_vector)
    feature_vector = np.reshape(feature_vector, (len(test_X_arr), 4))

    # Extracting feature-4: Interquartile Range
    IQR = []
    for i in range(len(test_X_arr)):
        Q1 = np.percentile(test_X_arr[i], 25, interpolation='midpoint')
        Q3 = np.percentile(test_X_arr[i], 75, interpolation='midpoint')
        IQR.append(Q3 - Q1)

    IQR = np.array(IQR)

    x = list()
    x.append(IQR)
    x = np.array(x)
    x = x.T

    feature_matrix = np.append(rolling_mean, rolling_std, axis=1)
    feature_matrix = np.append(feature_matrix, Fourier_frequency, axis=1)
    feature_matrix = np.append(feature_matrix, Fourier_peak, axis=1)
    feature_matrix = np.append(feature_matrix, polyfit_reg, axis=1)
    feature_matrix = np.append(feature_matrix, feature_vector, axis=1)
    feature_matrix = np.append(feature_matrix, x, axis=1)

    feature_matrix = np.nan_to_num(feature_matrix)
    # sc = StandardScaler()
    sc = MinMaxScaler()
    X_std = sc.fit_transform(feature_matrix)

    from sklearn.preprocessing import StandardScaler

    feature_matrix = StandardScaler().fit_transform(feature_matrix)
    df_feature_matrix = pd.DataFrame(feature_matrix)
    df_feature_matrix.fillna(0, inplace=True)

    from sklearn import preprocessing
    data_scaled = pd.DataFrame(preprocessing.scale(df_feature_matrix), columns=df_feature_matrix.columns)
    pca = decomposition.PCA(n_components=8)
    pca1 = decomposition.PCA(n_components=30)
    X_std_pca = pca.fit_transform(data_scaled)
    X_std_pca1 = pca1.fit_transform(data_scaled)

    pcaTransformed = pca.transform(feature_matrix)
    pcaTransformed1 = pca1.transform(feature_matrix)

    pcaTransformed = pd.DataFrame(pcaTransformed)

    X_test = np.asarray(pcaTransformed)


    # In[171]:


    # model = DBSCAN(eps=3, min_samples=5)
    model = DBSCAN(eps=2.5, min_samples=3, algorithm='kd_tree', metric='euclidean', leaf_size=90, p=2)
    dbscan_labels = model.fit_predict(train_X)
    core_samples = np.zeros_like(dbscan_labels, dtype = bool)
    core_samples[model.core_sample_indices_] = True

    # dbscan_labels=  model.labels_
    n_clusters_ = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    # print(n_clusters_)

    # colors = np.array([x for x in 'bgrcmywk'*10])
    colors = ['royalblue', 'maroon', 'forestgreen', 'mediumorchid', 'tan', 'deeppink', 'olive', 'goldenrod', 'lightcyan', 'navy']

    vectorizer = np.vectorize(lambda x: colors[x % len(colors)])
    plt.scatter(train_X[:, 0], train_X[:, 1],  c=vectorizer(dbscan_labels))

    new_train = []
    new_bins = []
    for i in range(len(dbscan_labels)):
        if dbscan_labels[i] == 0:
            new_train.append(train_X[i])
            new_bins.append(binned_weights[i])
    new_train = np.array(new_train)

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=6, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(new_train)
    plt.scatter(new_train[:, 0], new_train[:, 1], c=kmeans.labels_, s=50, cmap='gist_rainbow');

    y_pred1 = []
    y_pred1 = kmeans.labels_
    _dict = {}
    new_dict = {}

    for j in range(0, 6):
        for i in range(len(y_pred1)):
            if y_pred1[i] == j:
                if (new_bins[i] not in _dict):
                    _dict[new_bins[i]] = 1
                else:
                    _dict[new_bins[i]] += 1
        _max = 0
        #     print(_dict)
        for k in _dict.keys():
            if _dict[k] > _max:
                final_key = k
                _max = _dict[k]
        new_dict[j] = final_key
    # y = max(set(d), key = d.count)
    # d.sort()
    # print(d)
    # print(_dict)
    # print(new_dict)
    final_dict = {}
    bin_array = [0, 1, 2, 3, 4, 5]
    # print(bin_array)
    for i in new_dict.keys():
        if new_dict[i] in bin_array:
            bin_array.remove(new_dict[i])
            final_dict[i] = new_dict[i]
    # print(bin_array)
    # print(final_dict)
    for i in new_dict.keys():
        if i not in final_dict:
            final_dict[i] = bin_array[0]
            bin_array.remove(bin_array[0])
    # print(final_dict)
    # print("\n")

    # for i in range(len(y_pred1)):
    #     if y_pred1[i] == -1:
    #         y_pred1[i] = 7
    #     else:
    #         y_pred1[i] = final_dict[y_pred1[i]]

    # print(y_pred1)

    # print(y_pred1)


    # In[174]:


    from sklearn.metrics.pairwise import euclidean_distances
    dbscan_to_knn = {}
    for i in range(len(X_test)):
        distances = {}
        for j in range(len(new_train)):
            dist = np.linalg.norm(X_test[i]-new_train[j])
            # euc_dist = euclidean_distances(test_X[i],train_X[j])
            distances[j] = dist
        sorted_distances = sorted(distances.items(), key=lambda kv: kv[1])
        _keys = []
        for m in range(25):
            _keys.append(sorted_distances[m][0])
        # print(_keys)
        y_pred_labels = {}
        for l in _keys:
            y_pred_labels[l] = y_pred1[l]
    #     print(y_pred_labels)
        _list = list(y_pred_labels.values())
        final_key = max(set(_list ), key = _list.count)
        dbscan_to_knn[i] = final_key

    # print(dbscan_to_knn)


    # In[175]:

    dbscan_prediction = list(dbscan_to_knn.values())
    dbscan_prediction = np.array(dbscan_prediction)
    # print(dbscan_prediction)


    # In[176]:

    from sklearn import metrics
    score = metrics.accuracy_score(test_binned_weights[:51], dbscan_prediction)
    print("Score of DBSCAN Clustering: ", score)

    # features = X
    # np.savetxt("db_features.csv", features, delimiter=",")

    return dbscan_prediction
# In[177]:


