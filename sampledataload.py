from sklearn.cluster import AffinityPropagation
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
from sklearn.preprocessing import Imputer

# file = pd.read_csv("maokaiStats.csv", nrows=10000)
file = pd.read_csv("maokaiSupport.csv")
# file2 = pd.read_csv("maokaiSupport.csv", sep="\t")

# original = pd.DataFrame(file2, columns = ['id','position','win','kills','deaths','assists','totdmgdealt','magicdmgdealt','physicaldmgdealt','truedmgdealt','totdmgtaken','magicdmgtaken','physdmgtaken','truedmgtaken','totminionskilled','neutralminionskilled','ownjunglekills','enemyjunglekills'])


setcols = pd.DataFrame(file, columns=['id','position','win','kills','deaths','assists','totdmgdealt','magicdmgdealt','physicaldmgdealt','truedmgdealt','totdmgtaken','magicdmgtaken','physdmgtaken','truedmgtaken','totminionskilled','neutralminionskilled','ownjunglekills','enemyjunglekills'])
setcols2 = pd.DataFrame(file, columns=['id','position','win','kills','deaths','assists','totdmgdealt','magicdmgdealt','physicaldmgdealt','truedmgdealt','totdmgtaken','magicdmgtaken','physdmgtaken','truedmgtaken','totminionskilled','neutralminionskilled','ownjunglekills','enemyjunglekills'])
#Pick only support roles
# setcols = setcols.loc[setcols['position'] == "JUNGLE"]


columns = ['kills', 'deaths','assists','totdmgdealt','totdmgtaken']
setcols = setcols[columns]


X = setcols.values
labels_true = setcols.values

af = AffinityPropagation().fit(X)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_

n_clusters_ = len(cluster_centers_indices)

#####################
# save clusters
cluster_map = pd.DataFrame()
cluster_map['data_index'] = file.index.values
cluster_map['cluster'] = af.labels_

print('Length of clusters: %d' %n_clusters_)
# print(af.labels_)
# print(cluster_map[cluster_map.cluster==4])

######################
# Helper functions

# Returns array of indexes that belongs to a passed in cluster number
def clusterIndicesNumpy(clustNum, labels_array): #numpy
    return np.where(labels_array == clustNum)[0]

#display cluster index and composition
# cluster_to_check = 7
# print('##############################')
# print('Showing samples in cluster %d' %cluster_to_check)
# print(clusterIndicesNumpy(cluster_to_check, af.labels_))
# print('##############################')
# print('Showing row values of each data point in cluster %d' %cluster_to_check)
# print(X[clusterIndicesNumpy(cluster_to_check, af.labels_)])

#############################################
# GEt high winrate clusters
##########print(cluster_map[cluster_map.cluster==4])
from statistics import mean
from heapq import nlargest
curmeanlist = []
cindexbiggest = []
for i in range(0, n_clusters_):
    cur_cluster = cluster_map[cluster_map.cluster == i]
    if len(cur_cluster) >= 10:
        index = cur_cluster['data_index'].tolist()
        curmeanlist.append(mean(setcols2.iloc[index,2].tolist())) #2 = kills column

print(curmeanlist)
curmeanlist = np.array(curmeanlist)
cindexbiggest = nlargest(50, range(len(curmeanlist)), curmeanlist.take) #store top 10 biggest clusters in descending order
print(cindexbiggest)

from itertools import chain
#############################################
# Form scatter plot of highest win-rate clusters
masterlist = []
header_index_to_evaluate = 3 #'kills' = 0, 'deaths' = 1,'assists' = 2,'totdmgdealt' = 3,'totdmgtaken' = 4
x_axis = []
cl_num = 0
for i in range(0, len(cindexbiggest)):
    cur_cluster = X[clusterIndicesNumpy(cindexbiggest[i], af.labels_)] #retrieve array of values in specified cluster
    add_to_master = [item[header_index_to_evaluate] for item in cur_cluster]
    masterlist.append(add_to_master)
    for j in range(0, len(add_to_master)):
        x_axis.append(cl_num)
    cl_num += 1
y_axis = list(chain.from_iterable(masterlist))
mean = [np.mean(y_axis)]*len(x_axis)
# hard_mean = [0.155878494] * len(mean)
plt.figure(2)
plt.scatter(x_axis, y_axis, s=5, alpha=0.5)
# plt.plot(x_axis, hard_mean, label='Mean', linewidth=1.0, color="black")
plt.title('Total damage dealt by top 50 high win-rate maokai supports')
plt.xlabel('Cluster index')
plt.ylabel('Total damage dealt')


#Create list of clusters that contains value array for each

masterlist = []
x_axis = []
cl_num = 0
for i in range(0, n_clusters_) :
    cur_cluster = X[clusterIndicesNumpy(i, af.labels_)] #retrieve array of values in each cluster
    add_to_master = [item[header_index_to_evaluate] for item in cur_cluster] #retrieve only specified column from array
    masterlist.append(add_to_master)
    for j in range(0, len(add_to_master)):
        x_axis.append(cl_num)
    cl_num += 1

length_of_master = len(masterlist)
y_axis = list(chain.from_iterable(masterlist))

# print('y-axis length={}, x-axis length={}'.format(len(y_axis), len(x_axis)))
# print(y_axis)
# print(x_axis)

#############################################
# Form scatter plot of all clusters
plt.figure(1)
mean = [np.mean(y_axis)]*len(x_axis)
# hard_mean = [0.155878494] * len(mean)
# print(mean)
# print(hard_mean)
print("MEAN VALUE = {}".format(mean))
plt.scatter(x_axis, y_axis, s=5, alpha=0.5)
# plt.plot(x_axis, hard_mean, label='Mean', linewidth=1.0, color="black")
plt.title('Total damage dealt by maokai supports')
plt.xlabel('Cluster index')
plt.ylabel('Total damage dealt')
######################################
# Form cluster plot
# plt.close('all')
# plt.figure(2)
# plt.clf()
#
# colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
# for k, col in zip(range(n_clusters_), colors):
#     class_members = labels == k
#     cluster_center = X[cluster_centers_indices[k]]
#     plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
#     plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
#              markeredgecolor='k', markersize=10)
#     for x in X[class_members]:
#         plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)
#
# plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
#########################################