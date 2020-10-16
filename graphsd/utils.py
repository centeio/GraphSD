import math
import pandas as pd

from orangecontrib.associate.fpgrowth import *

class NominalSelector:
    def __init__(self, attribute, value):
        self.attribute = attribute
        self.value = value
    def __repr__(self):
        return "("+self.attribute+", "+ str(self.value)+")"
    def __str__(self):
        return "("+self.attribute+", "+ str(self.value)+")"
    def __eq__(self, other):
        return self.attribute == other.attribute and self.value == other.value
    def __lt__(self, other):
        if self.attribute != other.attribute:
            return self.attribute < other.attribute
        else:
            return self.value < other.value
    def __hash__(self):
        return hash(str(self))

class Pattern:
    def __init__(self, name, graph, weight): #name has to be of type list of NominalSelector
        self.name = name
        self.graph = graph
        self.weight = weight
        self.quality = 0
    def __repr__(self):
        return str(self.name)
    def __str__(self):
        return str(self.name)
    def __eq__(self, other):
        set(self.name) == set(other.name)
    def __lt__(self, other):
        return self.quality < other.quality

class NoGPattern:
    def __init__(self, name, pids, weight): #name has to be of type list of NominalSelector
        self.name = name
        self.weight = weight
        self.quality = 0
        self.ids = pids
    def __repr__(self):
        return str(self.name)
    def __str__(self):
        return str(self.name)
    def __eq__(self, other):
        set(self.name) == set(other.name)
    def __lt__(self, other):
        return self.quality < other.quality

def getBins(numBins, data):
    data_points_per_bin = math.ceil(len(data) / numBins)
    #listProSoc = list(socialData.ProSoc_z)
    sortedData = data.copy()
    sortedData.sort()
    
    binList = data.copy()
    
    i = 0
    limitinf = min(data)
    for el in range(len(data)):
        if data[el] == limitinf:
            binList[el] = 0
    
    for j in range(numBins):
        limitsup = sortedData[data_points_per_bin * (j + 1) - 1]
        for el in range(len(data)):
            if data[el] > limitinf and data[el] <= limitsup:
                binList[el] = j
                
        limitinf = limitsup

    return binList

def getBins2(numBins, data):
    data_points_per_bin = math.ceil(len(data) / numBins)
    #listProSoc = list(socialData.ProSoc_z)
    sortedData = data.copy()
    sortedData.sort()
    
    binList = data.copy()
    
    i = 0
    limitinf = min(data)
    for el in range(len(data)):
        if data[el] == limitinf:
            binList[el] = 0
    
    for j in range(numBins):
        pos = data_points_per_bin * (j + 1) - 1
        
        if pos >= len(sortedData):
            pos = len(sortedData) - 1
        limitsup = sortedData[pos]
        for el in range(len(data)):
            if data[el] > limitinf and data[el] <= limitsup:
                binList[el] = j
                
        limitinf = limitsup

    return binList



def getZ(val, mean, std):
    return (val - mean)/std


def addVelXY(dataframe):  # has to be sorted ascending by timestamp!!
    first = True
    ids = dataframe.id.unique()

    for i in ids:
        tempdf = dataframe.query("id == @i").copy()

        x1 = list(tempdf.x)
        x2 = list(tempdf.x)

        x2.pop(0)
        x2 += [x1[-1]]

        tempdf['velX'] = [px1 - px2 for (px1, px2) in zip(x2, x1)]

        y1 = list(tempdf.y)
        y2 = list(tempdf.y)
        y2.pop(0)
        y2 += [y1[-1]]

        tempdf['velY'] = [py1 - py2 for (py1, py2) in zip(y2, y1)]

        if first:
            resdf = tempdf.copy()
            first = False
        else:
            resdf = pd.concat([resdf, tempdf])

    return resdf

def freqItemsets(transactions, prun = 20):
    intTransactionsDict = {}
    last = 1
    intT = []

    for trans in transactions:
        temp = []
        for att in trans:
            if att not in intTransactionsDict:
                intTransactionsDict[att] = last
                last += 1
            temp += [intTransactionsDict[att]]
        intT += [temp]

    inv_intTransactionsDict = {v: k for k, v in intTransactionsDict.items()}

    itemsets = list(frequent_itemsets(intT, prun))

    newTransactions = {}
    for fset, count in itemsets:
        first = True
        for n in fset:
            if first:
                temp = (inv_intTransactionsDict[n],)
                first = False
            else:
                temp += (inv_intTransactionsDict[n],)

        newTransactions[temp] = count
        
    return newTransactions
