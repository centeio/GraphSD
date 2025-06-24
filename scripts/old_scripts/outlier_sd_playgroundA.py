import numpy as np
import pandas as pd
from scipy.spatial import distance
import math

from graphSD.outlier import *
from graphSD.viz import *
from graphSD.utils import *
from graphSD.sd import *

import pysubgroup as ps

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
    all_ids = socialData['id']

    cleandf.set_index('date', inplace=True)
    cleandf = cleandf[cleandf['id'].isin(list(socialData['id']))]

    ids = cleandf['id'].unique()

    # calculate speed

    newdf = addVelXY(cleandf)

    initialDate = '2016-10-10 11:30:01'
    finalDate = '2016-10-10 12:35:06'


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


    lof = getSeveralLOF(cleandf, socialData, initialDate, finalDate, nseconds=10, k = 3)
    areas = getSeveralAreas(cleandf, socialData, initialDate, finalDate, 10)

    #target = ps.NumericTarget('area')
    #searchspace = ps.create_selectors(areas, ignore=["area"])
    #task = ps.SubgroupDiscoveryTask (areas, target, searchspace, 
    #            depth=2, qf=ps.StandardQFNumeric(1))
    #result = ps.BeamSearch().execute(task)
    #for (q, sg) in result:
    #    print(str(q) + ":\t" + str(sg.subgroup_description) + ":\t" + str(sg.target.target_variable))

    #target = ps.NumericTarget('lof')
    #searchspace = ps.create_selectors(lof, ignore=["lof"])
    #task = ps.SubgroupDiscoveryTask (lof, target, searchspace, 
    #            depth=2, qf=ps.StandardQFNumeric(1))
    #result = ps.BeamSearch().execute(task)
    #for (q, sg) in result:
    #    print(str(q) + ":\t" + str(sg.subgroup_description) + ":\t" + str(sg.target.target_variable))


    attributes = ['Gender', 'Age_P', 'ProSoc_z_P', 'Conduct_z_P', 'Emotion_z_P', 'Peer_z_P', 'Hyper_z_P']

    transactions = setAtts(lof, attributes)
    itemsets = freqItemsets(transactions,1)

    tQ = treeQuality_nodes(lof, itemsets, 'lof')
    tQ.sort(reverse=True)
    infoPats_nodes(tQ,lof).to_csv('output/playgroundA_outlier_sd_lof.csv', index=True)

    transactions = setAtts(areas, attributes)
    itemsets = freqItemsets(transactions,1)

    tQ = treeQuality_nodes(areas, itemsets, 'area')
    tQ.sort(reverse=True)
    infoPats_nodes(tQ,areas).to_csv('output/playgroundA_outlier_sd_area.csv', index=True)
