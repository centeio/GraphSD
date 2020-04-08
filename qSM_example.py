import numpy as np
import pandas as pd
from scipy.spatial import distance
import math

from graphSD.graph import *
from graphSD.viz import *
from graphSD.utils import *
from graphSD.sd import *

np.random.seed(1234)

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

#### gather interactions

initialDate = '2016-10-10 11:50:01'
finalDate = '2016-10-10 12:00:06'

counter = getMultiDInteractions(cleandf, initialDate, finalDate, 0.5)

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

#### give attributes to graphs

attributes = ['Gender', 'Age_P', 'ProSoc_z_P', 'Conduct_z_P', 'Emotion_z_P', 'Peer_z_P', 'Hyper_z_P']

transactionsComp = setMultiCompAttEdges(GComp, socialData, attributes)
transactionsFrom = setMultiFromAttEdges(GFrom, socialData, attributes)
transactionsTo = setMultiToAttEdges(GTo, socialData, attributes)

#### Subgroup Discovery

compTQ = treeQuality(GComp,freqItemsets(transactionsComp, 10), qSM)
compTQ.sort(reverse=True)
infoPats(compTQ).to_csv('output/Comp_qSM_mean.csv', index=True)

compFrom = treeQuality(GFrom,freqItemsets(transactionsFrom, 10), qSM)
compFrom.sort(reverse=True)
infoPats(compFrom).to_csv('output/From_qSM_mean.csv', index=True)

compTo = treeQuality(GTo,freqItemsets(transactionsTo, 10), qSM)
compTo.sort(reverse=True)
infoPats(compTo).to_csv('output/To_qSM_mean.csv', index=True)

# using variance as metric

compTQ = treeQuality(GComp,freqItemsets(transactionsComp, 10), qSM, 'var')
compTQ.sort(reverse=True)
infoPats(compTQ).to_csv('output/Comp_qSM_var.csv', index=True)

compFrom = treeQuality(GFrom,freqItemsets(transactionsFrom, 10), qSM, 'var')
compFrom.sort(reverse=True)
infoPats(compFrom).to_csv('output/From_qSM_var.csv', index=True)

compTo = treeQuality(GTo,freqItemsets(transactionsTo, 10), qSM, 'var')
compTo.sort(reverse=True)
infoPats(compTo).to_csv('output/To_qSM_var.csv', index=True)