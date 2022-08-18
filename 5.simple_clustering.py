'''
source:
https://stackoverflow.com/questions/10136470/unsupervised-clustering-with-unknown-number-of-clusters

'''


import matplotlib.pyplot as plt
import numpy
import scipy.cluster.hierarchy as hcluster
import pandas as pd
from sklearn.preprocessing import StandardScaler
import math

# generate 3 clusters of each around 100 points and one orphan point
# N=100
# data = numpy.random.randn(3*N,2)
# data[:N] += 5
# data[-N:] += 10
# data[-1:] -= 20

data = pd.read_csv('/Users/beto/Documents/20_LAXFORD/pattsLength_10_b.csv')
# data = data.sample(n=500, random_state=0)
labels = data['futOut']
train = data.drop("futOut",axis=1)
train = train.drop("d3",axis=1)

# scaler = StandardScaler()
# print(scaler.fit(train))
# StandardScaler(copy=True, with_mean=True, with_std=True)
# dataSt = scaler.transform(train)
# dataSt = StandardScaler().fit_transform(train.values)


# clustering
thresh = 3#1.5
clusters = hcluster.fclusterdata(train, thresh, criterion="distance")

# plotting
plt.scatter(*numpy.transpose(train), c=clusters)
plt.axis("equal")
title = "threshold: %f, number of clusters: %d" % (thresh, len(set(clusters)))
plt.title(title)
plt.show()