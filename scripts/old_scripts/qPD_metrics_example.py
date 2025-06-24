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

    socialData = pd.read_csv('./data/useme2.csv', dtype={'id':str})

    cleandf = pd.read_csv('./data/cleandf.csv')
    cleandf = cleandf.rename(columns = {'Unnamed: 0':'date'})
    cleandf['id'] = cleandf['id'].astype(str)

    cleandf['date'] = pd.to_datetime(cleandf['date'])

    min_utc = cleandf.utc.min()
    #pd.to_datetime(min_utc,unit="ms") 

    max_utc = cleandf.utc.max()
    ids = socialData['id']

    cleandf.set_index('date', inplace=True)
    cleandf = cleandf[cleandf['id'].isin(list(socialData['id']))]

    # calculate speed

    newdf = addVelXY(cleandf)

    #### gather interactions

    initialDate = '2016-10-10 11:15:18'
    finalDate = '2016-10-10 12:44:13'

    counter = getDInteractions_all(newdf, initialDate, finalDate, 1)

    #### create graphs Comp, From and To

    GComp = createDiGraph(counter, ids)
    GTo = createDiGraph(counter, ids)
    GFrom = createDiGraph(counter, ids)

    #### PREPARE DATA

    ageMean = np.mean(socialData.AgeM)
    ageStd = np.std(socialData.AgeM)
    age_z = getZ(socialData.AgeM, ageMean, ageStd)

    socialData['Age_P'] = getBins(3,list(age_z))

    ProSocMean = np.mean(socialData.ProSoc_z)
    ProSocStd = np.std(socialData.ProSoc_z)
    ProSoc_z = getZ(socialData.ProSoc_z, ProSocMean, ProSocStd)

    socialData['ProSoc_z_P'] = getBins(3,list(socialData.ProSoc_z))

    socialData['Conduct_z_P'] = getBins(3,list(socialData.Conduct_z))

    socialData['Emotion_z_P'] = getBins(3,list(socialData.Emotion_z))

    socialData['Peer_z_P'] = getBins(3,list(socialData.Peer_z))

    socialData['Hyper_z_P'] = getBins(3,list(socialData.Hyper_z))

    ## metrics

    hubs, auths = nx.hits(GComp)
    degC = nx.centrality.degree_centrality(GComp)
    inDeg = nx.centrality.in_degree_centrality(GComp)
    outDeg = nx.centrality.out_degree_centrality(GComp)
    eigC = nx.centrality.eigenvector_centrality(GComp)
    closeness = nx.centrality.closeness_centrality(GComp)
    betweeness = nx.centrality.betweenness_centrality(GComp)
    pagerank = nx.pagerank(GComp)

    socialData['hubs'] = getBins(3,list(hubs.values())).copy()
    socialData['auths'] = getBins(3,list(auths.values())).copy()
    socialData['degC'] = getBins(3,list(degC.values())).copy()
    socialData['outDeg'] = getBins(3,list(outDeg.values())).copy()
    socialData['inDeg'] = getBins(3,list(inDeg.values())).copy()
    socialData['eigC'] = getBins(3,list(eigC.values())).copy()
    socialData['closeness'] = getBins(3,list(closeness.values())).copy()
    socialData['betweeness'] = getBins(3,list(betweeness.values())).copy()
    socialData['pagerank'] = getBins(3,list(pagerank.values())).copy()

    #### give attributes to graphs

    attributes = ["Gender", "Age_P", "hubs", "auths", "degC", "outDeg", "inDeg", "eigC", "closeness", "betweeness", "pagerank"]

    transactionsComp = setCompAttDiEdges(GComp, socialData, attributes)
    transactionsFrom = setFromAttDiEdges(GTo, socialData, attributes)
    transactionsTo = setToAttDiEdges(GFrom, socialData, attributes)

    #### Subgroup Discovery

    compTQ = treeQuality(GComp,freqItemsets(transactionsComp, 10), qP)
    compTQ.sort(reverse=True)
    infoPats(compTQ).to_csv('output/Comp_metrics_qPD.csv', index=True)

    compFrom = treeQuality(GFrom,freqItemsets(transactionsFrom, 10), qP)
    compFrom.sort(reverse=True)
    infoPats(compFrom).to_csv('output/From_metrics_qPD.csv', index=True)

    compTo = treeQuality(GTo,freqItemsets(transactionsTo, 10), qP)
    compTo.sort(reverse=True)
    infoPats(compTo).to_csv('output/To_metrics_qPD.csv', index=True)

    #### visualize

    #displayGender(GComp,ids, socialData, filtere=4)
    #graphViz(compTQ[0].graph)

    #TODO show example printpositions and printpositionsG
