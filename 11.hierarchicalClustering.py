'''
SOURCE:
https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/

'''

# needed imports
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import pandas as pd

np.set_printoptions(precision=5, suppress=True)



# generate two clusters: a with 100 points, b with 50:
np.random.seed(4711)  # for repeatability of this tutorial
# a = np.random.multivariate_normal([10, 0], [[3, 1], [1, 4]], size=[100,])
# b = np.random.multivariate_normal([0, 20], [[3, 1], [1, 4]], size=[50,])
# X = np.concatenate((a, b),)


#data = pd.read_csv('/Users/beto/Documents/20_LAXFORD/pattsLength_03_b.csv')
# labels = data['futOut']
# train = data.drop("futOut",axis=1)
# X = train.drop("d3",axis=1)
# X=np.array(X)

data = pd.read_csv('/Users/beto/Documents/20_LAXFORD/pattsLength_15.csv')
X=np.array(data)
print(X.shape)  # 150 samples with 2 dimensions


plt.scatter(X[:,0], X[:,1])
plt.show()

# generate the linkage matrix
Z = linkage(X, 'ward')


from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

c, coph_dists = cophenet(Z, pdist(X))
c

