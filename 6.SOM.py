'''
source:
https://towardsdatascience.com/analyzing-climate-patterns-with-self-organizing-maps-soms-8d4ef322705b

code:
https://github.com/hhl60492/SOMPY_robust_clustering/blob/master/sompy/examples/main.py

custoomized SOMPY:
https://github.com/hhl60492/SOMPY_robust_clustering


TIPS:
this script works with python 2.7. In order to set it as defautl version,
must be set as project interpreter. (.virtualenvs6)
If error "cannot connect with console" then prove this solution:
----
In case somebody still needs a solution:
go to (settings - build - console - python console) and set "working directory" box to your working area directory,
----

'''

import numpy as np
from sompy.sompy import SOMFactory
import pandas as pd
import glob
import os



# read in all csvs from folder
path = '/Users/beto/Documents/20_LAXFORD/ClimateData'
all_files = glob.glob(os.path.join(path, "*.csv"))

# concat into one df
df_from_each_file = (pd.read_csv(f, skiprows = 31) for f in all_files)
concatenated_df   = pd.concat(df_from_each_file, ignore_index=True)

# get columns Lat, Long, Mean Temp, Max Temp, Min temp, Precipitation
data = concatenated_df[['Lat', 'Long', 'Tm', 'Tx', 'Tn', 'P']]
data = data.apply(pd.to_numeric,  errors='coerce')
data = data.dropna(how='any')
names = ['Latitude', "longitude", 'Monthly Median temperature (C)','Monthly Max temperature (C)', 'Monthly Min temperature (C)', 'Monthly total precipitation (mm)']

print(data.head())

# create the SOM network and train it. You can experiment with different normalizations and initializations
sm = SOMFactory().build(data.values, normalization = 'var', initialization='pca', component_names=names)
sm.train(n_job=1, verbose=False, train_rough_len=2, train_finetune_len=5)

# The quantization error: average distance between each data vector and its BMU.
# The topographic error: the proportion of all data vectors for which first and second BMUs are not adjacent units.
topographic_error = sm.calculate_topographic_error()
quantization_error = np.mean(sm._bmu[1])
print ("Topographic error = %s; Quantization error = %s" % (topographic_error, quantization_error))

# component planes view
# from sompy.visualization.mapview import View2D
# view2D  = View2D(10,10,"rand data",text_size=12)
# view2D.show(sm, col_sz=4, which_dim="all", desnormalize=True)

import sompy
v = sompy.mapview.View2DPacked(10,10,"rand data",text_size=12)
v.show(sm, which_dim="all", col_sz=4)

# U-matrix plot
from sompy.visualization.umatrix import UMatrixView

umat  = UMatrixView(width=10,height=10,title='U-matrix')
umat.show(sm)

# do the K-means clustering on the SOM grid, sweep across k = 2 to 20
from sompy.visualization.hitmap import HitMapView
K = 20 # stop at this k for SSE sweep
K_opt = 18 # optimal K already found
km = sm.cluster(20)
hits  = HitMapView(20,20,"Clustering",text_size=12)
a=hits.show(sm)

# ----------------------------------- more on plotting.. BEGIN
import gmplot

gmap = gmplot.GoogleMapPlotter(54.2, -124.875224, 6)
j = 0
for i in km.cluster_centers_:
    gmap.marker(i[0],i[1],'red', title="Centroid " + str(j))
    j += 1

gmap.draw("centroids_map.html")


from bs4 import BeautifulSoup

def insertapikey(fname, apikey):
    """put the google api key in a html file"""
    def putkey(htmltxt, apikey, apistring=None):
        """put the apikey in the htmltxt and return soup"""
        if not apistring:
            apistring = "https://maps.googleapis.com/maps/api/js?key=%s&callback=initMap"
        soup = BeautifulSoup(htmltxt, 'html.parser')
        body = soup.body
        src = apistring % (apikey, )
        tscript = soup.new_tag("script", src=src, async="defer")
        body.insert(-1, tscript)
        return soup
    htmltxt = open(fname, 'r').read()
    soup = putkey(htmltxt, apikey)
    newtxt = soup.prettify()
    open(fname, 'w').write(newtxt)
API_KEY= 'YOUR API KEY HERE'
insertapikey("centroids_map.html", API_KEY)


gmap = gmplot.GoogleMapPlotter(54.2, -124.875224, 6)
j = 0
for i in km.cluster_centers_:
    gmap.marker(i[0],i[1],'red', title="Centroid " + str(j))
    j += 1

gmap.draw("centroids_map.html")

# ----------------------------------- more on plotting.. END



'''
ADAPTATION FOR FINANCE
'''

import numpy as np
from sompy.sompy import SOMFactory
import pandas as pd
import glob
import os
# import sklearn.external.joblib as joblib
import joblib


data = pd.read_csv('/Users/beto/Documents/20_LAXFORD/pattsLength_15.csv')
print(data.head())
names = ['d1','d2','d3','d4','d5','d6','d7','d8','d9','d10','d11','d12','d13','d14','d15']

# create the SOM network and train it. You can experiment with different normalizations and initializations
sm = SOMFactory().build(data.values, normalization = 'var', initialization='pca', component_names=names)
sm.train(n_job=1, verbose=False, train_rough_len=2, train_finetune_len=5)

# The quantization error: average distance between each data vector and its BMU.
# The topographic error: the proportion of all data vectors for which first and second BMUs are not adjacent units.
topographic_error = sm.calculate_topographic_error()
quantization_error = np.mean(sm._bmu[1])
print ("Topographic error = %s; Quantization error = %s" % (topographic_error, quantization_error))

# component planes view
# from sompy.visualization.mapview import View2D
# view2D  = View2D(10,10,"rand data",text_size=12)
# view2D.show(sm, col_sz=4, which_dim="all", desnormalize=True)

import sompy
v = sompy.mapview.View2DPacked(10,10,"rand data",text_size=12)
v.show(sm, which_dim="all", col_sz=4)

# U-matrix plot
from sompy.visualization.umatrix import UMatrixView

umat  = UMatrixView(width=10,height=10,title='U-matrix')
umat.show(sm)

# do the K-means clustering on the SOM grid, sweep across k = 2 to 20
from sompy.visualization.hitmap import HitMapView
K = 20 # stop at this k for SSE sweep
K_opt = 18 # optimal K already found
km = sm.cluster(20)
hits  = HitMapView(20,20,"Clustering",text_size=12)
a=hits.show(sm)

# ----------------------------------- more on plotting.. BEGIN
import gmplot

gmap = gmplot.GoogleMapPlotter(54.2, -124.875224, 6)
j = 0
for i in km.cluster_centers_:
    gmap.marker(i[0],i[1],'red', title="Centroid " + str(j))
    j += 1

gmap.draw("centroids_map.html")


from bs4 import BeautifulSoup

def insertapikey(fname, apikey):
    """put the google api key in a html file"""
    def putkey(htmltxt, apikey, apistring=None):
        """put the apikey in the htmltxt and return soup"""
        if not apistring:
            apistring = "https://maps.googleapis.com/maps/api/js?key=%s&callback=initMap"
        soup = BeautifulSoup(htmltxt, 'html.parser')
        body = soup.body
        src = apistring % (apikey, )
        tscript = soup.new_tag("script", src=src, async="defer")
        body.insert(-1, tscript)
        return soup
    htmltxt = open(fname, 'r').read()
    soup = putkey(htmltxt, apikey)
    newtxt = soup.prettify()
    open(fname, 'w').write(newtxt)
API_KEY= 'YOUR API KEY HERE'
insertapikey("centroids_map.html", API_KEY)


gmap = gmplot.GoogleMapPlotter(54.2, -124.875224, 6)
j = 0
for i in km.cluster_centers_:
    gmap.marker(i[0],i[1],'red', title="Centroid " + str(j))
    j += 1

gmap.draw("centroids_map.html")