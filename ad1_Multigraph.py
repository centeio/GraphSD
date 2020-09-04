import numpy as np
import pandas as pd
from scipy.spatial import distance
import math

from graphSD.graph import *
from graphSD.viz import *
from graphSD.utils import *
from graphSD.sd import *

np.random.seed(1234)

if __name__ ==  '__main__':

    socialData = pd.read_csv('./data/socialData_ad2.csv', dtype={'id':str})

    cleandf = pd.read_csv('./data/Movement_ad2.csv')
    cleandf['id'] = cleandf['id'].astype(str)


    cleandf['date'] = pd.to_datetime(cleandf['date'])

    #pd.to_datetime(min_utc,unit="ms") 
    ids = socialData['id']

    cleandf.set_index('date', inplace=True)
    cleandf = cleandf[cleandf['id'].isin(list(socialData['id']))]

    # calculate speed

    newdf = addVelXY(cleandf)
    #### gather interactions

    initialDate = "2020-06-10 09:00"
    finalDate = "2020-06-10 09:19"

    counter = getMultiDInteractions(newdf, initialDate, finalDate, 1)

    GComp = createMultiDiGraph(counter, ids)
    GFrom = createMultiDiGraph(counter, ids)
    GTo = createMultiDiGraph(counter, ids)

    #### give attributes to graphs

    attributes = ['At1', 'At2', 'At3']

    transactionsComp = setMultiCompAttDiEdges(GComp, socialData, attributes)
    transactionsFrom = setMultiFromAttDiEdges(GFrom, socialData, attributes)
    transactionsTo = setMultiToAttDiEdges(GTo, socialData, attributes)

    #### Subgroup Discovery

    print(freqItemsets(transactionsComp, 1))
    compTQ = treeQuality(GComp,freqItemsets(transactionsComp, 10), qS)
    compTQ.sort(reverse=True)
    infoPats(compTQ).to_csv('output/AD2_Comp_qSM_mean.csv', index=True)

    compFrom = treeQuality(GFrom,freqItemsets(transactionsFrom, 10), qS)
    compFrom.sort(reverse=True)
    infoPats(compFrom).to_csv('output/AD2_From_qSM_mean.csv', index=True)

    compTo = treeQuality(GTo,freqItemsets(transactionsTo, 10), qS)
    compTo.sort(reverse=True)
    infoPats(compTo).to_csv('output/AD2_To_qSM_mean.csv', index=True)


    #### visualize

    #displayGender(GComp,ids, socialData, filtere=4)
    #graphViz(compTQ[0].graph)

    #TODO show example printpositions and printpositionsG
