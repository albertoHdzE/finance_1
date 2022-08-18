# # Let's start off by importing the relevant libraries
# import pandas as pd
# import numpy as np
# import math
# import itertools
# from sklearn.cluster import KMeans
# from sklearn.metrics import pairwise_distances_argmin
# from datetime import datetime
# import time
# import matplotlib.pyplot as plt

# # Import raw data
# def import_data():
#     raw_data_df = pd.read_csv("/Users/beto/PycharmProjects/trading/DATA/Power-Networks-LCL-June2015(withAcornGps)v2_1.csv", header=0) # creates a Pandas data frame for input value
#     return raw_data_df


# raw_data_df = import_data()
# print(raw_data_df.head())


# #display all data replace result by desired raw_data
# result=raw_data_df
# result['date']=pd.to_datetime(result['DateTime'])
# data=result.loc[:, ['KWH/hh (per half hour) ']]
# data = data.set_index(result.date)
# data['KWH/hh (per half hour) '] = pd.to_numeric(data['KWH/hh (per half hour) '],downcast='float',errors='coerce')
# print(data.head())

# data.plot()
# plt.show()

# print(data.dropna().describe())

# # WEEKLY
# weekly = data.resample('W').sum()
# weekly.plot(style=[':', '--', '-'])
# plt.show()

# # DAYLY
# daily = data.resample('D').sum()
# daily.rolling(30, center=True).sum().plot(style=[':', '--', '-'])
# plt.show()
# day= data.resample(rule='H').sum()
# print(day.head())

# # We can do the same thing for a daily summary and we can use a
# # groupby and mean function for hourly summary:
# # PER HALF HOUR
# day.rolling(30*24, center=True).sum().plot(style=[':', '--', '-'])
# plt.show()

# by_time = data.groupby(data.index.time).mean()
# hourly_ticks = 4 * 60 * 60 * np.arange(6)
# by_time.plot(xticks=hourly_ticks, style=[':', '--', '-'])
# plt.show()

# data.rolling(360).mean().plot(figsize=(20,10), linewidth=5, fontsize=20)
# plt.xlabel('Year', fontsize=20);
# plt.show()

# # -----------------------------
# # ADITIONAL EXPLORATIONS
# # -----------------------------
# df = raw_data_df.loc[:,['date', 'KWH/hh (per half hour) ']]
# df['KWH/hh (per half hour) ']=pd.to_numeric(df['KWH/hh (per half hour) '],errors='coerce')
# df = df.groupby(['date']).sum().reset_index()
# print(df.head())
# df.plot.line(x = 'date', y = 'KWH/hh (per half hour) ',  figsize=(18,9), linewidth=5, fontsize=20)
# plt.show()

# mon = df['date']
# temp= pd.DatetimeIndex(mon)
# month = pd.Series(temp.month)
# to_be_plotted  = df.drop(['date'], axis = 1)
# to_be_plotted = to_be_plotted.join(month)
# to_be_plotted.plot.scatter(x = 'KWH/hh (per half hour) ', y = 'date', figsize=(16,8), linewidth=5, fontsize=20)
# plt.show()

# # for trend analysis
# df['KWH/hh (per half hour) '].rolling(5).mean().plot(figsize=(20,10), linewidth=5, fontsize=20)
# plt.show()

# # For seasonal variations
# # This is a cycle that repeats over time, such as monthly or yearly.
# # for deeper explanation:
# #   https://machinelearningmastery.com/time-series-seasonality-with-python/
# df['KWH/hh (per half hour) '].diff(periods=30).plot(figsize=(20,10), linewidth=5, fontsize=20)
# plt.show()

# pd.plotting.autocorrelation_plot(df['KWH/hh (per half hour) '])
# plt.show()

# from statsmodels.graphics.tsaplots import plot_acf
# plot_acf(df['KWH/hh (per half hour) '])
# plt.show()

# plot_acf(df['KWH/hh (per half hour) '], lags=50)
# plt.show()

# pd.plotting.lag_plot(df['KWH/hh (per half hour) '])
# plt.show()

# # -------------------------
# # Modeling with Prophet
# # -------------------------
# # We rename the columns in our data to the correct format.
# # The Date column must be called ‘ds’ and the value column we want
# # to predict ‘y’. We used our daily summary data in the example below.
# df2 = daily
# df2.reset_index(inplace=True)
# # prophet requieres columns ds(Date) and y (value)
# df2 = df2.rename(columns={'date': 'ds', 'KWH/hh (per half hour) ': 'y'})

# print(df2.head())

# # Then we import prophet, create a model and fit to the data. In
# # prophet, the changepoint_prior_scale parameter is used to control
# # how sensitive the trend is to changes, with a higher value being
# # more sensitive and a lower value less sensitive. After experimenting
# # with a range of values, I set this parameter to 0.10 up from the
# # default value of 0.05

# import fbprophet
# df2_prophet= fbprophet.Prophet(changepoint_prior_scale=0.10)
# df2_prophet.fit(df2)


# # To make forecasts, we need to create what is called a future dataframe.
# # We specify the number of future periods to predict (two months in
# # our case) and the frequency of predictions (daily). We then make
# # predictions with the prophet model we created and the future dataframe.

# # Make a future dataframe for 2 months
# df2_forecast = df2_prophet.make_future_dataframe(periods=30*2, freq='D')
# # Make predictions
# df2_forecast = df2_prophet.predict(df2_forecast)

# # The future dataframe contains the estimated household consumption
# # for the next two months. We can visualize the prediction with a plot:
# # The black dots represent the actual values, the blue line indicates
# # the forecasted values, and the light blue shaded region is the
# # uncertainty.

# # As illustrated in the next figure, the region of uncertainty grows
# # as we move further out in the future because the initial uncertainty
# # propagates and grows over time.

# df2_prophet.plot(df2_forecast, xlabel = 'Date', ylabel = 'KWH')
# plt.title('simple test')
# plt.show()

# # ---------------------------------
# # PLOTTING TRENDS AND PATTERS
# # ---------------------------------
# # The yearly pattern is interesting as it seems to suggest that the
# # household consumption increases in fall and winter, and decreases
# # in spring and summer. Intuitively, this is exactly what we expected
# # to see. Looking at the weekly trend, it seems that there is more
# # consumption on Sunday than the other days of the week. Finally, the
# # overall trend suggests that the consumption increases for a year
# # before slowly declining. Further investigation is needed to try to
# # explain this trend.
# df2_prophet.plot_components(df2_forecast)
# plt.show()

# # ---------------------------
# # LSTM prediction
# # ---------------------------
# # The Long Short-Term Memory recurrent neural network has the promise
# # of learning long sequences of observations.
# # *************************************************************************
# # ********* http://colah.github.io/posts/2015-08-Understanding-LSTMs/
# # - A recurrent neural network can be thought of as multiple copies of the
# # same network, each passing a message to a successor.
# # - Recurrent neural networks are intimately related to sequences and lists.
# # - One of the appeals of RNNs is the idea that they might be able to connect
# # previous information to the present task, such as using previous video
# # frames might inform the understanding of the present frame
# # - RNNs are absolutely capable of handling such “long-term dependencies.”
# # A human could carefully pick parameters for them to solve toy problems
# # of this form. Sadly, in practice, RNNs don’t seem to be able to learn
# # them, but Long Short Term Memory networks (LSTMN) can.
# # - All recurrent neural networks have the form of a chain of repeating
# # modules of neural network
# # *************************************************************************

# # Let’s use our daily summary data once again.
# mydata=daily.loc[:, ['KWH/hh (per half hour) ']]
# mydata = mydata.set_index(daily.index)
# print(mydata.head())

# # LSTMs are sensitive to the scale of the input data, specifically when
# # the sigmoid or tanh activation functions are used. It’s generally a
# # good practice to rescale the data to the range of [0, 1] or [-1, 1],
# # also called normalizing. We can easily normalize the dataset using the
# # MinMaxScaler preprocessing class from the scikit-learn library
# #Use MinMaxScaler to normalize 'KWH/hh (per half hour) ' to range from 0 to 1
# from sklearn.preprocessing import MinMaxScaler
# values = mydata['KWH/hh (per half hour) '].values.reshape(-1,1)
# values = values.astype('float32')
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled = scaler.fit_transform(values)

# # Now we can split the ordered dataset into train and test datasets.
# # The code below calculates the index of the split point and separates
# # the data into the training datasets with 80% of the observations
# # that we can use to train our model, leaving the remaining 20% for
# # testing the model.
# train_size = int(len(scaled) * 0.8)
# test_size = len(scaled) - train_size
# train, test = scaled[0:train_size,:], scaled[train_size:len(scaled),:]
# print(len(train), len(test))

# # We can define a function to create a new dataset and use this
# # function to prepare the train and test datasets for modeling.
# def create_dataset(dataset, look_back=1):
#     dataX, dataY = [], []
#     for i in range(len(dataset) - look_back):
#         a = dataset[i:(i + look_back), 0]
#         dataX.append(a)
#         dataY.append(dataset[i + look_back, 0])
#     print(len(dataY))
#     return np.array(dataX), np.array(dataY)

# look_back = 2
# trainX, trainY = create_dataset(train, look_back)
# testX, testY = create_dataset(test, look_back)

# # The LSTM network expects the input data to be provided with a specific
# # array structure in the form of: [samples, time steps, features].
# # Our data is currently in the form [samples, features] and we are
# # framing the problem as two time steps for each sample. We can
# # transform the prepared train and test input data into the expected
# # structure as follows:
# trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
# testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# # --------------------------------------------------
# # design and fit our LSTM network for our example.
# # --------------------------------------------------
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# model = Sequential()
# model.add(LSTM(100, input_shape=(trainX.shape[1], trainX.shape[2])))
# model.add(Dense(1))
# model.compile(loss='mae', optimizer='adam')
# history = model.fit(trainX, trainY, epochs=300, batch_size=100, validation_data=(testX, testY), verbose=0, shuffle=False)

# # From the plot of loss, we can see that the model has comparable
# # performance on both train and test datasets.
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='test')
# plt.legend()
# plt.show()

# # In the next figure we see that LSTM did a quite good job of
# # fitting the test dataset.
# yhat = model.predict(testX)
# plt.plot(yhat, label='predict')
# plt.plot(testY, label='true')
# plt.legend()
# plt.show()


# from math import sqrt
# from sklearn.metrics import mean_squared_error
# yhat_inverse = scaler.inverse_transform(yhat.reshape(-1, 1))
# testY_inverse = scaler.inverse_transform(testY.reshape(-1, 1))
# rmse = sqrt(mean_squared_error(testY_inverse, yhat_inverse))
# print('Test RMSE: %.3f' % rmse)

# plt.plot(yhat_inverse, label='predict')
# plt.plot(testY_inverse, label='actual', alpha=0.5)
# plt.legend()
# plt.show()

# # --------------------------------
# # CLUSTERING
# # --------------------------------
# # we can also do clustering with our sample data. There are quite a
# # few different ways of performing clustering, but one way is to
# # form clusters hierarchically. You can form a hierarchy in two ways:
# # start from the top and split, or start from the bottom and merge. I
# # decided to look at the latter in this post.

# # Let’s start with the data, we simply import the raw data and add two
# # columns for the day of the year and the hour of the day
# raw_data_df['date']=pd.to_datetime(raw_data_df['DateTime'])
# raw_data_df['dy']=raw_data_df['date'].dt.dayofyear
# raw_data_df['heure']=raw_data_df['date'].dt.time
# data_2014=raw_data_df.loc[:, ['heure','dy','KWH/hh (per half hour) ']]
# temp=raw_data_df.loc[:, ['dy','KWH/hh (per half hour) ']]
# data_2014['KWH/hh (per half hour) ']=pd.to_numeric(data_2014['KWH/hh (per half hour) '],errors='coerce')
# temp=temp.set_index(data_2014.heure)
# temp=data_2014.pivot_table(index=['heure'],columns=['dy'] ,values=['KWH/hh (per half hour) '],fill_value=0)

# print(temp.head())

# temp.plot(figsize=(12, 12))
# plt.show()


# # we can watch some of the graphs
# temp.iloc[:,0].plot(x=temp.index.get_level_values)
# temp.iloc[:,1].plot(x=temp.index.get_level_values)
# temp.iloc[:,2].plot(x=temp.index.get_level_values)
# plt.show()

# plt.figure(figsize=(11, 10))
# colors = ['#D62728', '#2C9F2C', '#FD7F23', '#1F77B4', '#9467BD',
#           '#8C564A', '#7F7F7F', '#1FBECF', '#E377C2', '#BCBD27',
#           '#CD5C5C', "#FFB500"]

# for i, r in enumerate([0, 1, 2, 3, 4, 5, 6, 73, 250, 275, 300], 1):
#     plt.subplot(4, 4, i)
#     plt.plot(temp.iloc[:, r], color=colors[i], linewidth=2)
#     plt.xlabel('Heures')
#     plt.legend(loc='upper right')
#     plt.tight_layout()

# plt.show()

# # ------------------------------
# # Linkage and Dendrograms
# # ------------------------------
# # The linkage function takes the distance information and groups pairs
# # of objects into clusters based on their similarity. These newly
# # formed clusters are next linked to each other to create bigger clusters.
# # This process is iterated until all the objects in the original data set
# # are linked together in a hierarchical tree.
# # To do clustering on our data:

# from scipy.cluster.hierarchy import dendrogram, linkage
# Z = linkage(temp.iloc[:,0:365], 'ward')

# # What does the ‘ward’ mean there and how does this actually work?
# # As the scipy linkage docs tell us, ward is one of the methods that can
# # be used to calculate the distance between newly formed clusters. The
# # keyword ‘ward’ causes linkage function to use the Ward variance
# # minimization algorithm. Other common linkage methods like single,
# # complete, average, and different distance metrics such as euclidean,
# # manhattan, hamming, cosine are also available if you want to play
# # around with.
# # Now let’s have a look at what’s called a dendogram of this hierarchical
# # clustering. Dendrograms are hierarchical plots of clusters where the
# # length of the bars represents the distance to the next cluster centre.
# plt.figure(figsize=(25, 10))
# plt.title('Hierarchical Clustering Dendrogram')
# plt.xlabel('sample index')
# plt.ylabel('distance')
# dendrogram(
#     Z,
#     leaf_rotation=90.,  # rotates the x axis labels
#     leaf_font_size=8.,  # font size for the x axis labels
# )
# plt.show()

# # - On the x axis you see labels. If you don’t specify anything else
# #   (like me) they are the indices of your samples in X.
# # - On the y axis you see the distances (of the ward method in our case).
# # - horizontal lines are cluster merges
# # - vertical lines tell you which clusters/labels were part of merge
# #   forming that new cluster
# # - heights of the horizontal lines tell you about the distance that
# #   needed to be “bridged” to form the new cluster
# # Even with explanations, the previous dendogram is still not obvious.
# # We can cut a little bit to be able to take a better look at the data.

# plt.title('Hierarchical Clustering Dendrogram (truncated)')
# plt.xlabel('sample index')
# plt.ylabel('distance')
# dendrogram(
#     Z,
#     truncate_mode='lastp',  # show only the last p merged clusters
#     p=12,  # show only the last p merged clusters
#     show_leaf_counts=False,  # otherwise numbers in brackets are counts
#     leaf_rotation=90.,
#     leaf_font_size=12.,
#     show_contracted=True,  # to get a distribution impression in truncated branches
# )
# plt.show()


# # -------------------------
# # RETRIEVING CLUSTERS
# # -------------------------

# # knowing max_d
# from scipy.cluster.hierarchy import fcluster
# max_d = 6
# clusters = fcluster(Z, max_d, criterion='distance')
# print(clusters)

# # knowing k
# k=2
# clusters2 = fcluster(Z, k, criterion='maxclust')
# print(clusters2)


# fl = fcluster(Z,2,criterion='maxclust')
# print(fl)





# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# --------------------------------------------
# --------------------------------------------
#              ADAPTATION
# --------------------------------------------
# --------------------------------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------



# Let's start off by importing the relevant libraries
import pandas as pd
import numpy as np
import math
import itertools
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from datetime import datetime
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Import raw data
def import_data():
    #raw_data_df = pd.read_csv("/Users/beto/PycharmProjects/trading/DATA/clean-2-avg.csv", header=0) # creates a Pandas data frame for input value
    raw_data_df = pd.read_csv("/Users/beto/Documents/20_LAXFORD/CLEAN-2-avg.csv", header=0)
    return raw_data_df


raw_data_df = import_data()
print(raw_data_df.head())


#display all data replace result by desired raw_data
result=raw_data_df
result['date']=pd.to_datetime(result['quote_time'],dayfirst=True)
data=result.loc[:, ['avgPrice']]
data = data.set_index(result.date)
data = data[(data[["avgPrice"]] != 0).all(axis=1)]
data['avgPrice']=pd.to_numeric(data['avgPrice'],downcast='float',errors='coerce')

# -----------first plot, single
# print(data.head())
# data.plot()
# plt.xticks(rotation='vertical')
# plt.show()

# ----------- second plot, detailed
fig, ax = plt.subplots()
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
#ax.xaxis.set_major_locator(mdates.DayLocator(interval=10))
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
ax.plot(data.index, data['avgPrice'])
fig.autofmt_xdate()
ax.grid(True)
plt.show()

# ------------ statistical description
print(data.dropna().describe())

# WEEKLY
#weekly = data.resample('W').sum()
weekly = data.resample('W').mean()
weekly.plot(style=[':', '--', '-'])
plt.show()

# MONTLY (ROLLED BY 30 DAYS)
daily = data.resample('D').sum()
daily.rolling(30, center=True).sum().plot(style=[':', '--', '-'])
plt.show()


# We can do the same thing for a daily summary and we can use a
# groupby and mean function for hourly summary:
day= data.resample(rule='H').sum()
print(day.head(50))
day.rolling(30*24, center=True).sum().plot(style=[':', '--', '-'])
plt.show()

# PER HALF HOUR
by_time = data.groupby(data.index.time).mean()
hourly_ticks = 4 * 60 * 60 * np.arange(6)
by_time.plot(xticks=hourly_ticks, style=[':', '--', '-'])
plt.show()

data.rolling(360).mean().plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('date', fontsize=20);
plt.show()

# -----------------------------
# ADITIONAL EXPLORATIONS
# -----------------------------
df = raw_data_df.loc[:,['quote_time', 'avgPrice']]
df['avgPrice']=pd.to_numeric(df['avgPrice'],errors='coerce')
df['quote_time']=pd.to_datetime(result['quote_time'],dayfirst=True)
df = df.groupby(['quote_time']).sum().reset_index()
print(df.head())

df.plot.line(x = 'quote_time', y = 'avgPrice',  figsize=(18,9), linewidth=5, fontsize=20)
plt.show()

# --------------- distribution of dates by day (or month)
mon = df['quote_time']
temp= pd.DatetimeIndex(mon)
#month = pd.Series(temp.month)
month = pd.Series(temp.hour)
to_be_plotted  = df.drop(['quote_time'], axis = 1)
to_be_plotted = to_be_plotted.join(month)
to_be_plotted.plot.scatter(x = 'avgPrice', y = 'quote_time', figsize=(16,8), linewidth=5, fontsize=20)
plt.show()

# for trend analysis ROLLING
# -----------------
# It takes a window size of k at a time and perform some desired
# mathematical operation on it. A window of size k means k
# consecutive values at a time. In a very simple case all the ‘k’
# values are equally weighted.
df['avgPrice'].rolling(5).mean().plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.show()

# For seasonal variations
# This is a cycle that repeats over time, such as monthly or yearly.
# Calculates the difference of a DataFrame element compared
# with another element in the DataFrame (default is the
# element in the same column of the previous row). Here 30
# for deeper explanation:
#   https://machinelearningmastery.com/time-series-seasonality-with-python/
df['avgPrice'].diff(periods=30).plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.show()

# AUTOCORRELATION PLOTS
# The idea is that, for each lag ℎ, we go through the series and check whether
# the data point ℎ time steps away covaries positively or negatively (i.e. when
# 𝑡 goes above the mean of the series, does 𝑡+ℎ also go above or below?).

# Running following code creates a 2D plot showing the lag value along the x-axis
# and the correlation on the y-axis between -1 and 1.
# Confidence intervals are drawn as a cone. By default, this is set to a 95%
# confidence interval, suggesting that correlation values outside of this cone
# are very likely a correlation and not a statistical fluke.

# Positive correlation: Both variables move in the same direction. In other words,
# as one variable increases, the other variable also increases. As one variable
# decreases, the other variable also decreases.
# Negative correlation: The variables move in opposite directions. As one variable
# increases, the other variable decreases. As one variable decreases, the other
# variable increases.

# The strength of a correlation indicates how strong the relationship is between
# the two variables. The strength is determined by the numerical value of the
# correlation. A correlation of 1, whether it is +1 or -1, is a perfect correlation.
# In perfect correlations, the data points lie directly on the line of fit. The
# further the data are from the line of fit, the weaker the correlation. A
# correlation of 0 indicates that there is no correlation. The following should be
# considered when determining the strength of a correlation:
# * When comparing a positive correlation to a negative correlation, only look at
#   the numerical value. Do not consider whether or not the correlation is positive
#   or negative. The correlation with the highest numerical value is the strongest.
# * If your line of best fit is horizontal or vertical like the scatterplots on
#   the top row, or if you are unable to draw a line of best fit because there is
#   no pattern in the data points, then there is little or no correlation.


pd.plotting.autocorrelation_plot(df['avgPrice'])
plt.show()

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(df['avgPrice'])
plt.show()

plot_acf(df['avgPrice'], lags=200)
plt.show()

# If time series is random, such autocorrelations should be near zero for any and
# all time-lag separations. If time series is non-random then one or more of the
# autocorrelations will be significantly non-zero. The horizontal lines displayed
# in the plot correspond to 95% and 99% confidence bands. The dashed line is 99%
# confidence band.

# In probability theory and statistics, partial correlation measures the degree of
# association between two random variables, with the effect of a set of controlling
# random variables removed. If we are interested in finding whether or to what
# extent there is a numerical relationship between two variables of interest,
# using their correlation coefficient will give misleading results if there is
# another, confounding, variable that is numerically related to both variables
# of interest. This misleading information can be avoided by controlling for the
# confounding variable, which is done by computing the partial correlation
# coefficient. This is precisely the motivation for including other right-side
# variables in a multiple regression; but while multiple regression gives unbiased
# results for the effect size, it does not give a numerical value of a measure
# of the strength of the relationship between the two variables of interest.

# For example, if we have economic data on the consumption, income, and wealth of
# various individuals and we wish to see if there is a relationship between
# consumption and income, failing to control for wealth when computing a
# correlation coefficient between consumption and income would give a misleading
# result, since income might be numerically related to wealth which in turn might
# be numerically related to consumption; a measured correlation between
# consumption and income might actually be contaminated by these other correlations.
# The use of a partial correlation avoids this problem.

# Like the correlation coefficient, the partial correlation coefficient takes on a
# value in the range from –1 to 1. The value –1 conveys a perfect negative
# correlation controlling for some variables (that is, an exact linear relationship
# in which higher values of one variable are associated with lower values of the
# other); the value 1 conveys a perfect positive linear relationship, and the
# value 0 conveys that there is no linear relationship.
# https://en.wikipedia.org/wiki/Partial_correlation
plot_pacf(df['avgPrice'], lags=800)
plt.show()

# Lag plots are used to check if a data set or time series is random. Random
# data should not exhibit any structure in the lag plot. Non-random structure
# implies that the underlying data are not random.
pd.plotting.lag_plot(df['avgPrice'])
plt.show()

# -------------------------
# Modeling with Prophet
# -------------------------
# We rename the columns in our data to the correct format.
# The Date column must be called ‘ds’ and the value column we want
# to predict ‘y’. We used our daily summary data in the example below.
df2 = df #daily
df2.reset_index(inplace=True)
# prophet requieres columns ds(Date) and y (value)
df2 = df2.rename(columns={'quote_time': 'ds', 'avgPrice': 'y'})

print(df2.head())

# Then we import prophet, create a model and fit to the data. In
# prophet, the changepoint_prior_scale parameter is used to control
# how sensitive the trend is to changes, with a higher value being
# more sensitive and a lower value less sensitive. After experimenting
# with a range of values, I set this parameter to 0.10 up from the
# default value of 0.05

import prophet
df2_prophet= prophet.Prophet(changepoint_prior_scale=0.10)
df2_prophet.fit(df2)


# To make forecasts, we need to create what is called a future dataframe.
# We specify the number of future periods to predict (two months in
# our case) and the frequency of predictions (daily). We then make
# predictions with the prophet model we created and the future dataframe.

# Make a future dataframe for 2 months
#df2_forecast = df2_prophet.make_future_dataframe(periods=30*2, freq='D')
df2_forecast = df2_prophet.make_future_dataframe(periods=1, freq='D')
# Make predictions
df2_forecast = df2_prophet.predict(df2_forecast)

# The future dataframe contains the estimated household consumption
# for the next two months. We can visualize the prediction with a plot:
# The black dots represent the actual values, the blue line indicates
# the forecasted values, and the light blue shaded region is the
# uncertainty.

# As illustrated in the next figure, the region of uncertainty grows
# as we move further out in the future because the initial uncertainty
# propagates and grows over time.

df2_prophet.plot(df2_forecast, xlabel = 'Date', ylabel = 'avgPrice')
plt.title('Prophets Model')
plt.show()

# ---------------------------------
# PLOTTING TRENDS AND PATTERS
# ---------------------------------
# The yearly pattern is interesting as it seems to suggest that the
# household consumption increases in fall and winter, and decreases
# in spring and summer. Intuitively, this is exactly what we expected
# to see. Looking at the weekly trend, it seems that there is more
# consumption on Sunday than the other days of the week. Finally, the
# overall trend suggests that the consumption increases for a year
# before slowly declining. Further investigation is needed to try to
# explain this trend.
df2_prophet.plot_components(df2_forecast)
plt.show()

# ---------------------------
# LSTM prediction
# ---------------------------
# The Long Short-Term Memory recurrent neural network has the promise
# of learning long sequences of observations.
# *************************************************************************
# ********* http://colah.github.io/posts/2015-08-Understanding-LSTMs/
# - A recurrent neural network can be thought of as multiple copies of the
# same network, each passing a message to a successor.
# - Recurrent neural networks are intimately related to sequences and lists.
# - One of the appeals of RNNs is the idea that they might be able to connect
# previous information to the present task, such as using previous video
# frames might inform the understanding of the present frame
# - RNNs are absolutely capable of handling such “long-term dependencies.”
# A human could carefully pick parameters for them to solve toy problems
# of this form. Sadly, in practice, RNNs don’t seem to be able to learn
# them, but Long Short Term Memory networks (LSTMN) can.
# - All recurrent neural networks have the form of a chain of repeating
# modules of neural network
# *************************************************************************

# Let’s use our daily summary data once again.
mydata=df.loc[:, ['avgPrice']]
mydata = mydata.set_index(df.index)
print(mydata.head())

# LSTMs are sensitive to the scale of the input data, specifically when
# the sigmoid or tanh activation functions are used. It’s generally a
# good practice to rescale the data to the range of [0, 1] or [-1, 1],
# also called normalizing. We can easily normalize the dataset using the
# MinMaxScaler preprocessing class from the scikit-learn library
#Use MinMaxScaler to normalize 'KWH/hh (per half hour) ' to range from 0 to 1
from sklearn.preprocessing import MinMaxScaler
values = mydata['avgPrice'].values.reshape(-1,1)
values = values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# Now we can split the ordered dataset into train and test datasets.
# The code below calculates the index of the split point and separates
# the data into the training datasets with 80% of the observations
# that we can use to train our model, leaving the remaining 20% for
# testing the model.
train_size = int(len(scaled) * 0.8)
test_size = len(scaled) - train_size
train, test = scaled[0:train_size,:], scaled[train_size:len(scaled),:]
print(len(train), len(test))

# We can define a function to create a new dataset and use this
# function to prepare the train and test datasets for modeling.
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)

look_back = 2
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# The LSTM network expects the input data to be provided with a specific
# array structure in the form of: [samples, time steps, features].
# Our data is currently in the form [samples, features] and we are
# framing the problem as two time steps for each sample. We can
# transform the prepared train and test input data into the expected
# structure as follows:
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# --------------------------------------------------
# design and fit our LSTM network for our example.
# --------------------------------------------------
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
model = Sequential()
model.add(LSTM(100, input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
history = model.fit(trainX, trainY, epochs=100, batch_size=100, validation_data=(testX, testY), verbose=0, shuffle=False)

# From the plot of loss, we can see that the model has comparable
# performance on both train and test datasets.
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# In the next figure we see that LSTM did a quite good job of
# fitting the test dataset.
yhat = model.predict(testX)
plt.plot(yhat, label='predict')
plt.plot(testY, label='true')
plt.legend()
plt.show()


from math import sqrt
from sklearn.metrics import mean_squared_error
yhat_inverse = scaler.inverse_transform(yhat.reshape(-1, 1))
testY_inverse = scaler.inverse_transform(testY.reshape(-1, 1))
rmse = sqrt(mean_squared_error(testY_inverse, yhat_inverse))
print('Test RMSE: %.3f' % rmse)

plt.plot(yhat_inverse, label='predict')
plt.plot(testY_inverse, label='actual', alpha=0.5)
plt.legend()
plt.show()

# --------------------------------
# CLUSTERING
# --------------------------------
# we can also do clustering with our sample data. There are quite a
# few different ways of performing clustering, but one way is to
# form clusters hierarchically. You can form a hierarchy in two ways:
# start from the top and split, or start from the bottom and merge. I
# decided to look at the latter in this post.

# Let’s start with the data, we simply import the raw data and add two
# columns for the day of the year and the hour of the day
# ------------------------------- ORIGINAL CODE BEGIN
# raw_data_df['date']=pd.to_datetime(raw_data_df['quote_time'],dayfirst=True)
# raw_data_df['dy']=raw_data_df['date'].dt.dayofyear
# raw_data_df['heure']=raw_data_df['date'].dt.time
#
# data_2019=raw_data_df.loc[:, ['heure','dy','avgPrice']]
# temp=raw_data_df.loc[:, ['dy','avgPrice']]
# data_2019['avgPrice']=pd.to_numeric(data_2019['avgPrice'],errors='coerce')
# temp=temp.set_index([data_2019.dy,data_2019.heure])
#
# temp = temp.resample('120min', how='mean')
# temp = temp.resample('60min', on='TIME').mean()
# #temp=temp.set_index(data_2019.dy)
# #temp=data_2019.pivot_table(index=['heure'],columns=['dy'] ,values=['avgPrice'],fill_value=0)
# temp=data_2019.pivot_table(index=['dy'],columns=['heure'] ,values=['avgPrice'],fill_value=0)
# # print(temp.head())
# # temp.plot(figsize=(12, 12))
# plt.show()
# ------------------------------- ORIGINAL CODE END


# ----------------------------------------
# ---------------------- modification 1 begin
# ----------------------------------------
raw_data_df['date']=pd.to_datetime(raw_data_df['quote_time'],dayfirst=True)
raw_data_df['avgPrice']=pd.to_numeric(raw_data_df['avgPrice'],errors='coerce')

temp=raw_data_df
temp['dy']=temp['date'].dt.dayofyear
temp['heure']=temp['date'].dt.time


# mean by hour
temp= temp.pivot_table(index=temp['date'].dt.hour,
                     columns='dy',
                     values='avgPrice',
                     aggfunc='mean')

# take NaN data off
# temp = temp.apply(lambda x: pd.Series(x.dropna().values))
# probably better interpolate and non pronning
temp=temp.interpolate()
temp.plot()
plt.xlabel('hour of the day')
plt.ylabel('value')
plt.show()


# ----------------------------------------
# ---------------------- modification 1 end
# ----------------------------------------

# we can watch some of the graphs
temp.iloc[:,0].plot(x=temp.index.get_level_values)
temp.iloc[:,1].plot(x=temp.index.get_level_values)
temp.iloc[:,2].plot(x=temp.index.get_level_values)
temp.iloc[:,3].plot(x=temp.index.get_level_values)
plt.show()

plt.figure(figsize=(11, 10))
colors = ['#D62728', '#2C9F2C', '#FD7F23', '#1F77B4', '#9467BD',
          '#8C564A', '#7F7F7F', '#1FBECF', '#E377C2', '#BCBD27',
          '#CD5C5C', "#FFB500"]

for i, r in enumerate([0, 1, 2, 3], 1):
    plt.subplot(4, 4, i)
    plt.plot(temp.iloc[:, r], color=colors[i], linewidth=2)
    plt.xlabel('date/time')
    plt.legend(loc='upper right')
    plt.tight_layout()

plt.show()

# ------------------------------
# Linkage and Dendrograms
# ------------------------------
# The linkage function takes the distance information and groups pairs
# of objects into clusters based on their similarity. These newly
# formed clusters are next linked to each other to create bigger clusters.
# This process is iterated until all the objects in the original data set
# are linked together in a hierarchical tree.
# To do clustering on our data:

from scipy.cluster.hierarchy import dendrogram, linkage

column_means = temp.mean()
temp = temp.fillna(column_means)
# Z = linkage(temp.iloc[:,0:365], 'ward')
# in this case, row 13 remains with NaN data, then is ignored just taking
# first 12 rows
Z = linkage(temp.iloc[:,:], 'centroid')

# What does the ‘ward’ mean there and how does this actually work?
# As the scipy linkage docs tell us, ward is one of the methods that can
# be used to calculate the distance between newly formed clusters. The
# keyword ‘ward’ causes linkage function to use the Ward variance
# minimization algorithm. Other common linkage methods like single,
# complete, average, and different distance metrics such as euclidean,
# manhattan, hamming, cosine are also available if you want to play
# around with.
# Now let’s have a look at what’s called a dendogram of this hierarchical
# clustering. Dendrograms are hierarchical plots of clusters where the
# length of the bars represents the distance to the next cluster centre.
plt.figure(figsize=(20, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

# - On the x axis you see labels. If you don’t specify anything else
#   (like me) they are the indices of your samples in X.
# - On the y axis you see the distances (of the ward method in our case).
# - horizontal lines are cluster merges
# - vertical lines tell you which clusters/labels were part of merge
#   forming that new cluster
# - heights of the horizontal lines tell you about the distance that
#   needed to be “bridged” to form the new cluster
# Even with explanations, the previous dendogram is still not obvious.
# We can cut a little bit to be able to take a better look at the data.

plt.title('Hierarchical Clustering Dendrogram (truncated)')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=12,  # show only the last p merged clusters
    show_leaf_counts=False,  # otherwise numbers in brackets are counts
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,  # to get a distribution impression in truncated branches
)
plt.show()


# -------------------------
# RETRIEVING CLUSTERS
# -------------------------

# knowing max_d
from scipy.cluster.hierarchy import fcluster
max_d = 6
clusters = fcluster(Z, max_d, criterion='distance')
print(clusters)

# knowing k
k=2
clusters2 = fcluster(Z, k, criterion='maxclust')
print(clusters2)


fl = fcluster(Z,2,criterion='maxclust')
print(fl)


# -------------------------
# REGRESION ON CLUSTERS
# -------------------------

# transponing: this makes not sence?
# final=pd.DataFrame(temp).T
# final.plot()
# plt.show()

# linkage search for patters by row. Then, transposing makes
# a row having all values hour by hour for a day
final=pd.DataFrame(temp).T
# here avoid NaN values, that is why I start from column 1
# I wonder if here makes sence interpolate for avoiding NaN
Z = linkage(final.iloc[0:4,1:15], 'centroid')
plt.title('Hierarchical Clustering Dendrogram (truncated)')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=12,  # show only the last p merged clusters
    show_leaf_counts=False,  # otherwise numbers in brackets are counts
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,  # to get a distribution impression in truncated branches
)
plt.show()





cluster1=final.iloc[:,0:4]
cluster1.plot()
plt.show()

cluster2=final.iloc[:,7:13]
cluster2.plot()
plt.show()


cluster1['average'] = cluster1.mean(numeric_only=True, axis=1)
cluster1.plot()
plt.show()

cluster2['average'] = cluster2.mean(numeric_only=True, axis=1)
cluster2.plot()
plt.show()


y = cluster1['average'].values
#x = np.array([0.0, 1.0, 2.0, 3.0])
x = range(len(y))
modelC1 = np.poly1d(np.polyfit(x, y, 3))
xp = np.linspace(0, 3, 100)
_ = plt.plot(x, y, '.', xp, modelC1(xp))
plt.ylim(23,23.8)
plt.xlabel('Ordinal', fontsize=10);
plt.ylabel('Value', fontsize=10);

plt.show()


y2 = cluster2['average'].values
modelC2 = np.poly1d(np.polyfit(x, y2, 3))

xp2 = np.linspace(0, 3, 100)
_ = plt.plot(x, y2, '.', xp2, modelC2(xp2))
plt.ylim(22.8,23.7)
plt.xlabel('Ordinal', fontsize=10);
plt.ylabel('Value', fontsize=10);

plt.show()



import sympy
from sympy import S, symbols


y = cluster1['average'].values
#x=np.array([0.0, 1.0, 2.0, 3.0])
x=range(len(y))

p = np.polyfit(x, y, 3)
f = np.poly1d(p)

# calculate new x's and y's
x_new = np.linspace(0, 3, 100)
y_new = f(x_new)

x = symbols("x")
poly = sum(S("{:6.2f}".format(v))*x**i for i, v in enumerate(p[::-1]))
eq_latex = sympy.printing.latex(poly)

plt.plot(x_new, y_new, label="${}$".format(eq_latex))
plt.legend(fontsize="small")
plt.show()



#x=np.array([0.0, 1.0, 2.0, 3.0])
y = cluster2['average'].values
x=range(len(y))

p = np.polyfit(x, y, 3)
f = np.poly1d(p)

# calculate new x's and y's
x_new = np.linspace(0, 3, 100)
y_new = f(x_new)

x = symbols("x")
poly = sum(S("{:6.2f}".format(v))*x**i for i, v in enumerate(p[::-1]))
eq_latex = sympy.printing.latex(poly)

plt.plot(x_new, y_new, label="${}$".format(eq_latex))
plt.legend(fontsize="small")
plt.show()

# *************************************************************
# *************************************************************
# --------------------------------------------------------
#      WORKING ON DATASET 2
# --------------------------------------------------------
# *************************************************************
# *************************************************************
# Import raw data
def import_data2():
    raw_data_df2 = pd.read_csv("/Users/beto/Documents/20_LAXFORD/CLEAN-2-avg.csv", header=0) # creates a Pandas data frame for input value
    return raw_data_df2


raw_data_df2 = import_data2()


raw_data_df2['date']=pd.to_datetime(raw_data_df2['quote_time'],dayfirst=True)
raw_data_df2['avgPrice']=pd.to_numeric(raw_data_df2['avgPrice'],errors='coerce')

temp2=raw_data_df2
temp2['dy']=temp2['date'].dt.dayofyear
temp2['heure']=temp2['date'].dt.time


temp2= temp2.pivot_table(index=temp2['date'].dt.hour,
                     columns='dy',
                     values='avgPrice',
                     aggfunc='mean')

# must be from 9:30 to 15:50
# each column = a day
# each row = an hour
# plot: x= hour (index), y=values
aux=temp2.iloc[5:13,[2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,21,22,23,24,25,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]]
temp2=aux.interpolate()
temp2.plot(fontsize=10)

plt.xlabel('hour')
plt.ylabel('value')
plt.legend(fontsize=5)
plt.show()

# each column = an hour
# each row = a day
# plot: x= hour (index), y=values
temp2=temp2.T
temp2=temp2.interpolate()

# next plot does not say much
# temp2.plot()
# plt.xlabel('day')
# plt.ylabel('value')
# plt.show()

# columns = hour
# rows = days
# it considers one row = on pattern, then makes clasiffication of rows
# 49 patterns classfied in 2 clusters
Z = linkage(temp2.iloc[:,0:7], 'centroid')

plt.figure(figsize=(20, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()



plt.title('Hierarchical Clustering Dendrogram (truncated)')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=12,  # show only the last p merged clusters
    show_leaf_counts=False,  # otherwise numbers in brackets are counts
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,  # to get a distribution impression in truncated branches
)
plt.show()

# ELBOW METHOD
from sklearn import preprocessing
from sklearn.cluster import KMeans
dataset1=temp2.iloc[:,0:8]
dataset1_standardized = preprocessing.scale(dataset1)
dataset1_standardized = pd.DataFrame(dataset1_standardized)

# find the appropriate cluster number
plt.figure(figsize=(10, 8))

wcss = []
for i in range(1, 8):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(dataset1_standardized)
    wcss.append(kmeans.inertia_)
plt.plot(range(0, 7), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()



Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans
score = [kmeans[i].fit(dataset1).score(dataset1) for i in range(len(kmeans))]
score
plt.plot(Nc, score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()


from scipy.cluster.hierarchy import fcluster
import sympy
from sympy import S, symbols
# fcluster can be used to flatten the dendrogram, obtaining as a
# result an assignation of the original data points to single
# clusters.
max_d = .5
clusters2 = fcluster(Z, max_d, criterion='distance')
print(clusters2)
k=max(clusters2)

# knowing k
k=11
clusters2 = fcluster(Z, k, criterion='maxclust')
print(clusters2)

avgClusters = pd.DataFrame()
for x in range(1, k+1):
    countClusters = x
    cluster1=temp2.iloc[np.where(clusters2==countClusters)[0].tolist(),:]
    plt.rcParams.update({'font.size': 2})
    cluster1=cluster1.T
    plt.subplot(9, 5, x)
    plt.plot(cluster1)
    plt.title('cluster ' + str(countClusters))
    #plt.xlabel('hour')
    #plt.ylabel('value')
    cluster1['avg'] = cluster1.mean(numeric_only=True, axis=1)
    avgClusters['cl'+str(x)]=cluster1['avg']
    plt.plot(cluster1['avg'],linewidth=1,c='cyan')

plt.show()


# plotting only average clusters with regression model

allEq=[]      # hold string models
allEqModel=[] # hold model objects
for j in range(0, k):
    xx=np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    y = avgClusters.iloc[:,j].values
    p = np.polyfit(xx, y, 5)
    f = np.poly1d(p)
    allEqModel.append(f)
    # calculate new x's and y's
    #x_new = np.linspace(0, 8, 100)
    y_new = f(xx)
    x = symbols("x")
    poly = sum(S("{:6.2f}".format(v))*x**i for i, v in enumerate(p[::-1]))
    eq_latex = sympy.printing.latex(poly)
    allEq.append(eq_latex)
    plt.subplot(9, 5, j+1)
    plt.plot(xx, y, '.')
    plt.plot(xx, y_new, label="${}$".format(eq_latex))
    plt.legend(fontsize="small")
    print(allEq[j-1])

plt.show()

# 222222222222222222222222222222222222222222222
# 222222222222222222222222222222222222222222222
# ---------------------------------------------
#       working with second data set
# ---------------------------------------------
# 222222222222222222222222222222222222222222222
# 222222222222222222222222222222222222222222222


# Import raw data
def import_data():
    raw_data_df = pd.read_csv("/Users/beto/Documents/20_LAXFORD/CLEAN-1-avg.csv", header=0) # creates a Pandas data frame for input value
    return raw_data_df


raw_data_df = import_data()

#display all data replace result by desired raw_data
result=raw_data_df
result['date']=pd.to_datetime(result['quote_time'],dayfirst=True)
result['avgPrice']=pd.to_numeric(result['avgPrice'],downcast='float',errors='coerce')

temp=result
temp['dy']=temp['date'].dt.dayofyear
temp['heure']=temp['date'].dt.time

temp= temp.pivot_table(index=temp['date'].dt.hour,
                     columns='dy',
                     values='avgPrice',
                     aggfunc='mean')

tempW=temp.iloc[5:13,:]


from sklearn import preprocessing
from scipy.spatial import distance

selectedModel=-1
futCounter = 0


for y in range(0,len(list(tempW))):
    futCounter = futCounter + 1
    min_max_scaler = preprocessing.MinMaxScaler()
    counter = 0
    dist = 1000000
    for ff in allEqModel:
        counter=counter+1
        y_new=ff(xx)
        y_new=y_new.reshape(-1,1)
        y_new = min_max_scaler.fit_transform(y_new)
        y_new = pd.DataFrame(y_new)
        plt.subplot(9, 5, counter)

        original = tempW.iloc[:,y].values
        original=original.reshape(-1,1)
        original = min_max_scaler.fit_transform(original)
        original = pd.DataFrame(original)
        plt.plot(xx, original)
        plt.plot(xx, y_new, label="${}$".format(eq_latex))
        plt.legend(fontsize="small")

        distAux = distance.euclidean(y_new.values.flatten(),original.values.flatten())
        if distAux < dist:
            dist=distAux
            selectedModel=counter-1


    plt.show()

    y_new = allEqModel[selectedModel](xx)
    y_new = y_new.reshape(-1, 1)
    y_new = min_max_scaler.fit_transform(y_new)
    y_new = pd.DataFrame(y_new)
    plt.plot(xx,original,'.')
    plt.plot(xx,y_new,'-',label="${}$".format(allEq[selectedModel]))
    plt.title('Future pattern ' + str(futCounter) + ' match past pattern ' + str(selectedModel) +
              ' with error= ' + str(dist),fontsize=10)
    plt.show()















# rows = hour
# columns = day
# plot() makes one line per column, then graph a day behaviour along
# the hour
temp3 = temp2.T

cluster1=temp3.iloc[:,[15,44,18,42,43]]
cluster1.plot()
plt.xlabel('hour')
plt.ylabel('value')
plt.show()

# obtaining model by regression in average of cluster1
cluster1['average'] = cluster1.mean(numeric_only=True, axis=1)

import sympy
from sympy import S, symbols

xx=np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
y = cluster1['average'].values
p = np.polyfit(xx, y, 7)
f = np.poly1d(p)
# calculate new x's and y's
x_new = np.linspace(0, 9, 100)
y_new = f(x_new)
x = symbols("x")
poly = sum(S("{:6.2f}".format(v))*x**i for i, v in enumerate(p[::-1]))
eq_latex = sympy.printing.latex(poly)
plt.plot(xx, y, '.')
plt.plot(x_new, y_new, label="${}$".format(eq_latex))
plt.plot(cluster1)
plt.legend(fontsize="small")
plt.show()



cluster2=temp3.iloc[:,[15,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47]]
cluster2.plot()
plt.xlabel('hour')
plt.ylabel('value')
plt.show()

# rescaling cluster 1
# I thought normalization to cluster1 i was going to see
# homogeneous patterns, but no, it is chaotic. Probably
# this is not a good idea coz a pattern is a function of
# its original scale
# from sklearn.preprocessing import MinMaxScaler
# from sklearn import preprocessing
#
# x = cluster1.values #returns a numpy array
# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(x)
# df = pd.DataFrame(x_scaled)
# df.plot()
# plt.show()




# take NaN data type out
#plt.plot(temp2['date'].dt.hour, temp2[2], linestyle='-', marker='o')
# temp2 = temp2.apply(lambda x: pd.Series(x.dropna().values))
# temp2=temp2.interpolate()
# temp2.plot()
# plt.show()

# -------------------------------------------------
# -------------------------------------------------





from scipy.cluster.hierarchy import dendrogram, linkage
#Z = linkage(temp.iloc[:,0:365], 'ward')
# in this case, row 13 remains with NaN data, then is ignored just taking
# first 12 rows
Z = linkage(temp2.iloc[0:15,0:50], 'centroid')

# What does the ‘ward’ mean there and how does this actually work?
# As the scipy linkage docs tell us, ward is one of the methods that can
# be used to calculate the distance between newly formed clusters. The
# keyword ‘ward’ causes linkage function to use the Ward variance
# minimization algorithm. Other common linkage methods like single,
# complete, average, and different distance metrics such as euclidean,
# manhattan, hamming, cosine are also available if you want to play
# around with.
# Now let’s have a look at what’s called a dendogram of this hierarchical
# clustering. Dendrograms are hierarchical plots of clusters where the
# length of the bars represents the distance to the next cluster centre.
plt.figure(figsize=(20, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

# - On the x axis you see labels. If you don’t specify anything else
#   (like me) they are the indices of your samples in X.
# - On the y axis you see the distances (of the ward method in our case).
# - horizontal lines are cluster merges
# - vertical lines tell you which clusters/labels were part of merge
#   forming that new cluster
# - heights of the horizontal lines tell you about the distance that
#   needed to be “bridged” to form the new cluster
# Even with explanations, the previous dendogram is still not obvious.
# We can cut a little bit to be able to take a better look at the data.

plt.title('Hierarchical Clustering Dendrogram (truncated)')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=12,  # show only the last p merged clusters
    show_leaf_counts=False,  # otherwise numbers in brackets are counts
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,  # to get a distribution impression in truncated branches
)
plt.show()


# -------------------------
# RETRIEVING CLUSTERS
# -------------------------

# knowing max_d
from scipy.cluster.hierarchy import fcluster
max_d = 6
clusters = fcluster(Z, max_d, criterion='distance')
print(clusters)

# knowing k
k=2
clusters2 = fcluster(Z, k, criterion='maxclust')
print(clusters2)


fl = fcluster(Z,2,criterion='maxclust')
print(fl)


# -------------------------
# REGRESION ON CLUSTERS
# -------------------------

final=pd.DataFrame(temp2).T
final.plot()
plt.show()

cluster1=final.iloc[:,[0,3,1,2,8,14,4,5,6]]
cluster1.plot()
plt.show()

cluster2=final.iloc[:,7]
cluster2.plot()
plt.show()


cluster1['average'] = cluster1.mean(numeric_only=True, axis=1)
cluster1.plot()
plt.show()

x = np.array([0.0, 1.0, 2.0, 3.0])
y = cluster1['average'].values
modelC1 = np.poly1d(np.polyfit(x, y, 3))
xp = np.linspace(0, 3, 100)
_ = plt.plot(x, y, '.', xp, modelC1(xp))
plt.ylim(23,23.8)
plt.xlabel('Ordinal', fontsize=10);
plt.ylabel('Value', fontsize=10);
plt.show()

cluster2['average'] = cluster2.mean(numeric_only=True, axis=1)
cluster2.plot()
plt.show()

y2 = cluster2['average'].values
modelC2 = np.poly1d(np.polyfit(x, y2, 3))

xp2 = np.linspace(0, 3, 100)
_ = plt.plot(x, y2, '.', xp2, modelC2(xp2))
plt.ylim(22.8,23.7)
plt.xlabel('Ordinal', fontsize=10);
plt.ylabel('Value', fontsize=10);

plt.show()



import sympy
from sympy import S, symbols

x=np.array([0.0, 1.0, 2.0, 3.0])
y = cluster1['average'].values

p = np.polyfit(x, y, 3)
f = np.poly1d(p)

# calculate new x's and y's
x_new = np.linspace(0, 3, 100)
y_new = f(x_new)

x = symbols("x")
poly = sum(S("{:6.2f}".format(v))*x**i for i, v in enumerate(p[::-1]))
eq_latex = sympy.printing.latex(poly)

plt.plot(x_new, y_new, label="${}$".format(eq_latex))
plt.legend(fontsize="small")
plt.show()



x=np.array([0.0, 1.0, 2.0, 3.0])
y = cluster2['average'].values

p = np.polyfit(x, y, 3)
f = np.poly1d(p)

# calculate new x's and y's
x_new = np.linspace(0, 3, 100)
y_new = f(x_new)

x = symbols("x")
poly = sum(S("{:6.2f}".format(v))*x**i for i, v in enumerate(p[::-1]))
eq_latex = sympy.printing.latex(poly)

plt.plot(x_new, y_new, label="${}$".format(eq_latex))
plt.legend(fontsize="small")
plt.show()