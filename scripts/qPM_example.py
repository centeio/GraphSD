from graphsd.patterns import *
from scripts.process_data import load_data

np.random.seed(1234)

transactionsComp, transactionsFrom, transactionsTo = load_data()

freqComp = freqItemsets(transactionsComp, 100)
print('len of freq',len(freqComp))

compTQ = treeQuality(GComp,freqComp, qP,  samples=100)
compTQ.sort(reverse=True)
infoPats(compTQ).to_csv('output/Comp_qPM_mean.csv', index=True)

compFrom = treeQuality(GFrom,freqItemsets(transactionsFrom, 100), qP,  samples=100)
compFrom.sort(reverse=True)
infoPats(compFrom).to_csv('output/From_qPM_mean.csv', index=True)

compTo = treeQuality(GTo,freqItemsets(transactionsTo, 100), qP,  samples=100)
compTo.sort(reverse=True)
infoPats(compTo).to_csv('output/To_qPM_mean.csv', index=True)

# using variance as metric

compTQ = treeQuality(GComp,freqItemsets(transactionsComp, 100), qP, metric = 'var',  samples = 100)
compTQ.sort(reverse=True)
infoPats(compTQ).to_csv('output/Comp_qPM_var.csv', index=True)

compFrom = treeQuality(GFrom,freqItemsets(transactionsFrom, 100), qP, metric = 'var',  samples = 100)
compFrom.sort(reverse=True)
infoPats(compFrom).to_csv('output/From_qPM_var.csv', index=True)

compTo = treeQuality(GTo,freqItemsets(transactionsTo, 100), qP, metric = 'var',  samples = 100)
compTo.sort(reverse=True)
infoPats(compTo).to_csv('output/To_qPM_var.csv', index=True)
