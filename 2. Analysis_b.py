'''
source:
http://www.aprendemachinelearning.com/k-means-en-python-paso-a-paso/

'''



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['figure.figsize'] = (10, 10)
plt.style.use('ggplot')


#dataframe = pd.read_csv("/Users/beto/PycharmProjects/trading/DATA/analisis.csv")
# _b coitains future output as last column
#dataframe = pd.read_csv("/Users/beto/Documents/20_LAXFORD/pattsLength_03_b.csv")
dataframe = pd.read_csv("/Users/beto/Documents/20_LAXFORD/pattsLength_10_b.csv")
dataframe.head().T
#Descriptive stats of the variables in data
desc=dataframe.describe()
print(desc)


#print(dataframe.groupby('categoria').size())
uniqueValues = dataframe.groupby('futOut').size().keys()
print(dataframe.groupby('futOut').size())
print(uniqueValues)



#dataframe.drop(['categoria'], 1).hist()
dataframe.drop(['futOut'], 1).hist()
plt.show()

# En este caso seleccionamos 3 dimensiones: op, ex y ag y las cruzamos para
# ver si nos dan alguna pista de su agrupación y la relación con sus categorías.
# dropna option = Drop missing values from the data before plotting.
# hue = 'categoría' = deferencía por el valor de la columna categoría
#sb.pairplot(dataframe.dropna(), hue='categoria',size=4,vars=["op","ex","ag"],kind='scatter')
plt.rcParams['figure.figsize'] = (10, 10)
sb.pairplot(dataframe.dropna(),vars=["d1","d2","d3","d4","d5","d6","d7","d8","d9","d10"],kind='scatter')
#sb.pairplot(dataframe.dropna(), size=4,vars=["d1","d2","d3"],kind='scatter')
plt.show()

# X = np.array(dataframe[["op","ex","ag"]])
# y = np.array(dataframe['categoria'])
# X.shape

X1 = np.array(dataframe[["d1","d2","d3"]])
X2 = np.array(dataframe[["d4","d5","d6"]])
X3 = np.array(dataframe[["d7","d8","d9"]])
X4 = np.array(dataframe[["d10","d1","d3"]])
X5 = np.array(dataframe[["d5","d7","d9"]])
X6 = np.array(dataframe[["d10","d2","d4"]])
y = np.array(dataframe['futOut'])


#values=[-100,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,-0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1]
y_copy = y
counter = len(uniqueValues)
alphabet = []
for letter in range(97,97+counter):
    alphabet.append(chr(letter))

counter=-1
for value in uniqueValues:
    counter +=1
    y_copy=np.where(y == value, alphabet[counter], y_copy)


counter=-1
for letter in alphabet:
    counter +=1
    y=np.where(y_copy == letter, counter, y)

y = y.astype(int)

fig = plt.figure()
ax = Axes3D(fig)
colores=['blue','red','green','blue','cyan','yellow','orange','black','pink','brown','purple','dimgray','saddlebrown','navy','salmon','brown','orchid','hotpink','tomato']
asignar=[]
for row in y:
    asignar.append(colores[row])
ax.scatter(X1[:, 0], X1[:, 1], X1[:, 2], c=asignar,s=60)

plt.show()


# Obtener el valor K
# ----------------------
# Vamos a hallar el valor de K haciendo una gráfica e intentando hallar el
# “punto de codo” que comentábamos antes. Este es nuestro resultado:
Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans
score = [kmeans[i].fit(X1).score(X1) for i in range(len(kmeans))]
score
plt.plot(Nc, score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()


# Ejecutamos K-Means
# ---------------------
# Ejecutamos el algoritmo para 5 clusters y obtenemos las etiquetas y
# los centroids

kmeans = KMeans(n_clusters=4).fit(X1)
centroids = kmeans.cluster_centers_
print(centroids)

# Ahora veremos esto en una gráfica 3D con colores para los grupos y
# veremos si se diferencian: (las estrellas marcan el centro de cada cluster)

# Predicting the clusters
labels = kmeans.predict(X1)
# Getting the cluster centers
C = kmeans.cluster_centers_
colores=['red','green','blue','cyan','yellow']
asignar=[]
for row in labels:
    asignar.append(colores[row])

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X1[:, 0], X1[:, 1], X1[:, 2], c=asignar,s=60)
ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c=colores, s=1000)

plt.show()


# Aqui podemos ver que el Algoritmo de K-Means con K=5 ha agrupado a los
# 140 usuarios Twitter por su personalidad, teniendo en cuenta las 3
# dimensiones que utilizamos: Openess, Extraversion y Agreeablenes.
# Pareciera que no hay necesariamente una relación en los grupos con sus
# actividades de Celebrity.
#
# Haremos 3 gráficas en 2 dimensiones con las proyecciones a partir de
# nuestra gráfica 3D para que nos ayude a visualizar los grupos y
# su clasificación:

# Getting the values and plotting it
# f1 = dataframe['op'].values
# f2 = dataframe['ex'].values
f1 = dataframe['d1'].values
f2 = dataframe['d2'].values

plt.scatter(f1, f2, c=asignar, s=70)
plt.scatter(C[:, 0], C[:, 1], marker='*', c=colores, s=1000)
plt.show()


# Getting the values and plotting it
# f1 = dataframe['op'].values
# f2 = dataframe['ag'].values
f1 = dataframe['d2'].values
f2 = dataframe['d3'].values

plt.scatter(f1, f2, c=asignar, s=70)
plt.scatter(C[:, 0], C[:, 2], marker='*', c=colores, s=1000)
plt.show()


# f1 = dataframe['ex'].values
# f2 = dataframe['ag'].values

f1 = dataframe['d3'].values
f2 = dataframe['d1'].values

plt.scatter(f1, f2, c=asignar, s=70)
plt.scatter(C[:, 1], C[:, 2], marker='*', c=colores, s=1000)
plt.show()



