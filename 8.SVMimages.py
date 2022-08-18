'''
taken from
https://www.kaggle.com/asparago/unsupervised-learning-with-som/data
'''


#As usual we start importing a number of libraries that will be come in handy later on
import numpy as np
import pandas as pd
import seaborn as sns
from imageio import imwrite
#from scipy.misc import imsave
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
#%matplotlib inline

import plotly.offline as py
#py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

from sklearn.preprocessing import StandardScaler
from PIL import Image, ImageChops
import SimpSOM as sps





np.random.seed(0)

#We load here the data from the provided training set, we randomly select 500 landmark points and separate the labels.
# train = pd.read_csv('/Users/beto/PycharmProjects/trading/DATA/train.csv')
train = pd.read_csv('/Users/beto/Documents/20_LAXFORD/pattsLength_10_b.csv')

train = train.sample(n=500, random_state=0)
# labels = train['label']
# train = train.drop("label",axis=1)
labels = train['futOut']
train = train.drop("futOut",axis=1)

#Let's plot the distribution and see if the distribution is uniform
sns.set(color_codes=True)
sns.distplot(labels.values,bins=np.arange(-0.5,10.5,1))
plt.show()




#Then we normalize the data, a crucial step to the correct functioning of the
# SOM algorithm
trainSt = StandardScaler().fit_transform(train.values)


#We build a 40x40 network and initialise its weights with PCA
net = sps.somNet(40, 40, trainSt, PBC=True, PCI=True)

#Now we can train it with 0.1 learning rate for 10000 epochs
net.train(0.1, 10000)

#We print to screen the map of the weights differences between nodes,
# this will help us identify cluster centers
net.diff_graph(show=True,printout=True)