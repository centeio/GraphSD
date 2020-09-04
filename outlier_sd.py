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

    initialDate = "2020-06-10 09:09"
    finalDate = "2020-06-10 09:19"

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

    attributes = ['At1', 'At2', 'At3', 'id']

    transactions = setAtts(lof, attributes)
    itemsets = freqItemsets(transactions,1)

    tQ = treeQuality_nodes(lof, itemsets, 'lof')
    tQ.sort(reverse=True)
    infoPats_nodes(tQ,lof).to_csv('output/outlier_sd_lof.csv', index=True)

    transactions = setAtts(areas, attributes)
    itemsets = freqItemsets(transactions,1)

    tQ = treeQuality_nodes(areas, itemsets, 'area')
    tQ.sort(reverse=True)
    infoPats_nodes(tQ,areas).to_csv('output/outlier_sd_area.csv', index=True)



    