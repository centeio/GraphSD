from graphsd.patterns import *
from graphsd.utils import *

np.random.seed(1234)

if __name__ == '__main__':
    socialData = pd.read_csv('./data/useme2.csv', dtype={'id': str})

    cleandf = pd.read_csv('./data/cleandf.csv')
    cleandf = cleandf.rename(columns={'Unnamed: 0': 'date'})
    cleandf['id'] = cleandf['id'].astype(str)

    cleandf['date'] = pd.to_datetime(cleandf['date'])

    min_utc = cleandf.utc.min()
    # pd.to_datetime(min_utc,unit="ms")

    max_utc = cleandf.utc.max()
    all_ids = socialData['id']

    cleandf.set_index('date', inplace=True)
    cleandf = cleandf[cleandf['id'].isin(list(socialData['id']))]

    ids = cleandf['id'].unique()

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

    socialData['Age_P'] = getBins(3, list(age_z))

    ProSocMean = np.mean(socialData.ProSoc_z)
    ProSocStd = np.std(socialData.ProSoc_z)
    ProSoc_z = getZ(socialData.ProSoc_z, ProSocMean, ProSocStd)

    socialData['ProSoc_z_P'] = getBins(3, list(socialData.ProSoc_z))

    socialData['Conduct_z_P'] = getBins(3, list(socialData.Conduct_z))

    socialData['Emotion_z_P'] = getBins(3, list(socialData.Emotion_z))

    socialData['Peer_z_P'] = getBins(3, list(socialData.Peer_z))

    socialData['Hyper_z_P'] = getBins(3, list(socialData.Hyper_z))

    #### give attributes to graphs

    attributes = ['Gender', 'Age_P', 'ProSoc_z_P', 'Conduct_z_P', 'Emotion_z_P', 'Peer_z_P', 'Hyper_z_P']

    transactionsComp = setCompAttDiEdges(GComp, socialData, attributes)
    transactionsFrom = setFromAttDiEdges(GFrom, socialData, attributes)
    transactionsTo = setToAttDiEdges(GTo, socialData, attributes)

    #### Subgroup Discovery

    compTQ = treeQuality(GComp, freqItemsets(transactionsComp, 10), qP)
    compTQ.sort(reverse=True)
    infoPats(compTQ).to_csv('output/Comp_qPD_mean.csv', index=True)

    compFrom = treeQuality(GFrom, freqItemsets(transactionsFrom, 10), qP)
    compFrom.sort(reverse=True)
    infoPats(compFrom).to_csv('output/From_qPD_mean.csv', index=True)

    compTo = treeQuality(GTo, freqItemsets(transactionsTo, 10), qP)
    compTo.sort(reverse=True)
    infoPats(compTo).to_csv('output/To_qPD_mean.csv', index=True)

    # Using variance as metric

    compTQ = treeQuality(GComp, freqItemsets(transactionsComp, 10), qP, metric='var')
    compTQ.sort(reverse=True)
    infoPats(compTQ).to_csv('output/Comp_qPD_var.csv', index=True)

    compFrom = treeQuality(GFrom, freqItemsets(transactionsFrom, 10), qP, metric='var')
    compFrom.sort(reverse=True)
    infoPats(compFrom).to_csv('output/From_qPD_var.csv', index=True)

    compTo = treeQuality(GTo, freqItemsets(transactionsTo, 10), qP, metric='var')
    compTo.sort(reverse=True)
    infoPats(compTo).to_csv('output/To_qPD_var.csv', index=True)

    #### visualize

    # displayGender(GComp,ids, socialData, filtere=4)
    # graphViz(compTQ[0].graph)

    # TODO show example printpositions and printpositionsG
