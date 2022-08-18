'''

'''

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import numpy as np
from numpy import loadtxt
import time
import functools
import csv
import pandas as pd

totalStart = time.time()

def convert_date(date_bytes):
    return mdates.strpdate2num('%d/%m/%Y %H:%M')(date_bytes.decode('ascii'))

# date, bid, ask = np.loadtxt('/Users/beto/Documents/20_LAXFORD/ML Patterns trading/DATA/GBPUSD1d.csv', unpack=True,
#                             delimiter=',',
#                             converters={0: mdates.strpdate2num('%Y%m%d%H%M%S')})

# date, bid, ask = np.loadtxt('/Users/beto/Documents/20_LAXFORD/sample_data_1_CLEAN3.csv', unpack=True,
#                             delimiter=',',
#                             converters={0: mdates.strpdate2num('%d/%m/%Y %H:%M')})

# date, bid, ask = np.loadtxt('/Users/beto/Documents/20_LAXFORD/sample_data_1_CLEAN3.csv', unpack=True,
#                             delimiter=',',
#                             converters={0: convert_date})

dataset = pd.read_csv('/Users/beto/Documents/20_LAXFORD/sample_data_1_CLEAN3.csv')
dataset["date"] = pd.to_datetime(dataset['date']).dt.date

date = dataset['date']
bid = dataset['bid']
ask = dataset['ask']


def percentChange(startPoint, currentPoint):
    try:
        x = ((float(currentPoint) - startPoint) / abs(startPoint)) * 100.00
        if x == 0.0:
            return 0.000000001
        else:
            return x
    except:
        return 0.0001


# dist2Patt: number of points far from current patter to start compute outcome
# lengthOutcome: number of points to compute outcome
def patternStorage(pattLength, dist2Patt, lengthOutcomePatt):
    '''
    The goal of patternFinder is to begin collection of %change patterns
    in the tick data. From there, we also collect the short-term outcome
    of this pattern. Later on, the length of the pattern, how far out we
    look to compare to, and the length of the compared range be changed,
    and even THAT can be machine learned to find the best of all 3 by
    comparing success rates.'''

    startTime = time.time()

    # here is: avgLine = allData[:toWhat]
    x = len(avgLine) - pattLength
    y = pattLength + 1
    currentStance = 'none'


    while y < x:
        pattern = []
        counter = pattLength

        while counter > 0:
            counter -=1
            pp = percentChange(avgLine[y - pattLength], avgLine[y - counter])
            pattern.append(pp)

        # originally, dist2Patt = 20, lengthOutcomePatt = 10, this way:
        # outcomeRange = avgLine[y + 20:y + 30]
        outcomeRange = avgLine[y + dist2Patt:y + (dist2Patt + lengthOutcomePatt)]
        currentPoint = avgLine[y]

        try:
            avgOutcome = functools.reduce(lambda x, y: x + y, outcomeRange) / len(outcomeRange)
        except Exception as e:
            print(str(e))
            avgOutcome = 0
        futureOutcome = percentChange(currentPoint, avgOutcome)

        # adding futureOutcome to the end of pattern
        pattern.append(str(round(futureOutcome, 1)))

        patternAr.append(pattern)
        performanceAr.append(futureOutcome)

        y += 1

    f = open("/Users/beto/Documents/20_LAXFORD/pattsLength_03_b.csv", "w")
    w = csv.writer(f)
    w.writerows(patternAr)
    f.close()

    endTime = time.time()
    print('patternArrays length is: ',str(len(patternAr)))
    print('performanceArrays length: ',str(len(performanceAr)))
    print('Pattern storing took:', endTime - startTime)


def currentPattern(pattLength, avgLine):
    mostRecentPoint = avgLine[1]

    counter = pattLength + 1
    while counter > 1:
        counter -=1
        cpp = percentChange(avgLine[-(pattLength + 1)], avgLine[-counter])
        patForRec.append(cpp)




def graphRawFX():
    fig = plt.figure(figsize=(10, 7))
    ax1 = plt.subplot2grid((40, 40), (0, 0), rowspan=40, colspan=40)
    ax1.plot(date, bid)
    ax1.plot(date, ask)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
    plt.grid(True)
    for label in ax1.xaxis.get_ticklabels():
        label.set_rotation(45)
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    ax1_2 = ax1.twinx()
    ax1_2.fill_between(date, 0, (ask - bid), facecolor='g', alpha=.3)

    plt.subplots_adjust(bottom=.23)
    plt.show()


def patternRecognition(pattLength, dist2Patt, lengthOutcomePatt):
    plotPatAr = []
    patFound = 0


    for eachPattern in patternAr:
        counter = -1
        howSim = 0
        while counter < pattLength - 1:
            counter += 1
            sim = 100.00 - abs(percentChange(eachPattern[counter], patForRec[counter]))
            howSim = howSim + sim

        howSim = howSim/pattLength

        if howSim > 70:
            patdex = patternAr.index(eachPattern)
            patFound = 1
            xp = range(1, pattLength + 1)
            #############

            plotPatAr.append(eachPattern)

    if patFound == 1:
        fig = plt.figure(figsize=(10, 6))

        for eachPatt in plotPatAr:
            futurePoints = patternAr.index(eachPatt)

            if performanceAr[futurePoints] > patForRec[9]:
                pcolor = '#24bc00'
            else:
                pcolor = '#d40000'

            plt.plot(xp, eachPatt)
            ####################
            plt.scatter(pattLength + 5, performanceAr[futurePoints], c=pcolor, alpha=.4)
        realOutcomeRange = allData[toWhat + dist2Patt:toWhat + (dist2Patt + lengthOutcomePatt)]
        realAvgOutcome = functools.reduce(lambda x, y: x + y, realOutcomeRange) / len(realOutcomeRange)
        realMovement = percentChange(allData[toWhat], realAvgOutcome)
        plt.scatter(pattLength+10, realMovement, c='#54fff7', s=25)
        plt.plot(xp, patForRec, '#54fff7', linewidth=3)
        plt.grid(True)
        plt.title(
            'Pattern Recognition.'
            '\nCyan line is the current pattern. Other lines are similar patterns from the past.'
            '\nPredicted outcomes are color-coded to reflect a positive or negative prediction.'
         #   '\nThe Cyan dot marks where the pattern went.'
         #   '\nOnly data in the past is used to generate patterns and predictions.'
            )
        plt.show()



dataLength = int(bid.shape[0])
print('data length is', dataLength)

allData = ((bid + ask) / 2)


toWhat = int(dataLength/2)
toWhat = int(dataLength-1)
pattLength = 3
dist2Patt = 10
lengthOutcomePatt = 3

while toWhat < dataLength:
    #avgLine = ((bid + ask) / 2)
    avgLine = allData[:toWhat]

    patternAr = []
    performanceAr = []
    patForRec = []

    # avgOutcome = reduce(lambda x, y: x + y, outcomeRange) / len(outcomeRange)

    patternStorage(pattLength, dist2Patt, lengthOutcomePatt)
    currentPattern(pattLength, avgLine)
    patternRecognition(pattLength, dist2Patt, lengthOutcomePatt)
    totalEnd = time.time() - totalStart
    print('Entire processing took:', totalEnd, 'seconds')
    toWhat += 1

