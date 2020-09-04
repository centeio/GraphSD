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

    counter = getMultiDInteractions(newdf, initialDate, finalDate, 1)

    GComp = createMultiDiGraph(counter, ids)
    GFrom = createMultiDiGraph(counter, ids)
    GTo = createMultiDiGraph(counter, ids)

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

    #### Create signed graph

    GLikes = nx.DiGraph()
    GLikes.add_nodes_from(ids)

    for n in list(GLikes.nodes()):
        node1 = socialData[socialData.id == n]['Dislike 1'].item()
        node2 = socialData[socialData.id == n]['Dislike 2'].item()
        node3 = socialData[socialData.id == n]['Dislike 3'].item()
        
        id1 = socialData[socialData.ID == (node1)].id.item()
        id2 = socialData[socialData.ID == (node2)].id.item()
        id3 = socialData[socialData.ID == (node3)].id.item()
        
        GLikes.add_edge(n,id1)
        GLikes.add_edge(n,id2)
        GLikes.add_edge(n,id3)


    #### give attributes to graphs

    attributes = ['Gender', 'Age_P', 'ProSoc_z_P', 'Conduct_z_P', 'Emotion_z_P', 'Peer_z_P', 'Hyper_z_P']

    transactionsComp = setMultiCompAttDiEdges(GComp, socialData, attributes, GLikes)
    transactionsFrom = setMultiFromAttDiEdges(GFrom, socialData, attributes, GLikes)
    transactionsTo = setMultiToAttDiEdges(GTo, socialData, attributes, GLikes)

    #### Subgroup Discovery

    compTQ = treeQuality(GComp,freqItemsets(transactionsComp, 100), qS, samples=100)
    compTQ.sort(reverse=True)
    infoPats(compTQ).to_csv('output/Comp_signed_qSM.csv', index=True)

    compFrom = treeQuality(GFrom,freqItemsets(transactionsFrom, 100), qS, samples=100)
    compFrom.sort(reverse=True)
    infoPats(compFrom).to_csv('output/From_signed_qSM.csv', index=True)

    compTo = treeQuality(GTo,freqItemsets(transactionsTo, 100), qS, samples=100)
    compTo.sort(reverse=True)
    infoPats(compTo).to_csv('output/To_signed_qSM.csv', index=True)