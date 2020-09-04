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

    socialData = pd.read_csv('./data/socialData_playgroundB.csv', dtype={'id':str})

    movement = pd.read_csv('./data/child14_11_14.csv', names = ['id', 'date', 'x', 'y'], header = None)
    movement['date'] = pd.to_datetime(movement['date'])
    movement.set_index('date', inplace=True)
    movement = movement.sort_index()
    movement = movement.groupby('id').resample('S').mean().reset_index()

    movement.set_index('date', inplace=True)
    movement = movement.sort_index()

    ids = movement.id.unique()

    socialData = pd.read_csv('./data/socialData.csv',header = 0, sep =';')
    socialData.reset_index()
    socialData = socialData[socialData['id'].isin(ids)]


    # calculate speed

    movement = addVelXY(movement)

    #### gather interactions

    initialDate = "2014-11-14 09:04"
    finalDate = "2014-11-14 09:56"

    counter = getMultiDInteractions_all(movement, initialDate, finalDate, 1)

    GComp = createMultiDiGraph(counter, ids)
    GFrom = createMultiDiGraph(counter, ids)
    GTo = createMultiDiGraph(counter, ids)

    #print(counter)
    #graphViz(GComp)

    #### PREPARE DATA

    #ageMean = np.mean(socialData.AgeM)
    #ageStd = np.std(socialData.AgeM)
    #age_z = getZ(socialData.AgeM, ageMean, ageStd)
    #socialData['AgeZ'] = age_z
    socialData['Age_P'] = getBins2(3,list(socialData['AgeM'])).copy()

    ## metrics

    #hubs, auths = nx.hits(GComp)
    #degC = nx.centrality.degree_centrality(GComp)
    #inDeg = nx.centrality.in_degree_centrality(GComp)
    #outDeg = nx.centrality.out_degree_centrality(GComp)
    #eigC = nx.centrality.eigenvector_centrality(GComp)
    #closeness = nx.centrality.closeness_centrality(GComp)
    #betweeness = nx.centrality.betweenness_centrality(GComp)
    #pagerank = nx.pagerank(GComp)

    #print(degC,inDeg,outDeg,eigC)

    #socialData['hubs'] =  getBins2(3,list(hubs.values())).copy()
    #socialData['auths'] = getBins2(3,list(auths.values())).copy()
    #socialData['degC'] = getBins2(3,list(degC.values())).copy()
    #socialData['outDeg'] = getBins2(3,list(outDeg.values())).copy()
    #socialData['inDeg'] = getBins2(3,list(inDeg.values())).copy()
    #socialData['eigC'] = getBins2(3,list(eigC.values())).copy()
    #socialData['closeness'] = getBins2(3,list(closeness.values())).copy()
    #socialData['betweeness'] = getBins2(3,list(betweeness.values())).copy()
    #socialData['pagerank'] = getBins2(3,list(pagerank.values())).copy()

    #print(socialData)

    #### give attributes to graphs

    attributes = ["Gender", "Age_P"]
    transactionsComp = setMultiCompAttDiEdges(GComp, socialData, attributes)
    transactionsFrom = setMultiFromAttDiEdges(GFrom, socialData, attributes)
    transactionsTo = setMultiToAttDiEdges(GTo, socialData, attributes)

    #### Subgroup Discovery

    compTQ = treeQuality(GComp,freqItemsets(transactionsComp, 1), qP, samples = 100)
    compTQ.sort(reverse=True)
    infoPats(compTQ).to_csv('output/playgroundB_Comp_qPM_mean.csv', index=True)

    compFrom = treeQuality(GFrom,freqItemsets(transactionsFrom, 1), qP, samples = 100)
    compFrom.sort(reverse=True)
    infoPats(compFrom).to_csv('output/playgroundB_From_qPM_mean.csv', index=True)

    compTo = treeQuality(GTo,freqItemsets(transactionsTo, 1), qP, samples = 100)
    compTo.sort(reverse=True)
    infoPats(compTo).to_csv('output/playgroundB_To_qPM_mean.csv', index=True)

    #var
    
    compTQ = treeQuality(GComp,freqItemsets(transactionsComp, 1), qP, metric = 'var', samples = 100)
    compTQ.sort(reverse=True)
    infoPats(compTQ).to_csv('output/playgroundB_Comp_qPM_var.csv', index=True)

    compFrom = treeQuality(GFrom,freqItemsets(transactionsFrom, 1), qP, metric = 'var', samples = 100)
    compFrom.sort(reverse=True)
    infoPats(compFrom).to_csv('output/playgroundB_From_qPM_var.csv', index=True)

    compTo = treeQuality(GTo,freqItemsets(transactionsTo, 1), qP, metric = 'var', samples = 100)
    compTo.sort(reverse=True)
    infoPats(compTo).to_csv('output/playgroundB_To_qPM_var.csv', index=True)