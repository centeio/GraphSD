intFromTransactionsDict = {}
lastFrom = 1

def getIntFromTransaction(transaction):
    global intFromTransactionsDict
    global lastFrom
    if transaction in intFromTransactionsDict:
        return intFromTransactionsDict[transaction]
    intFromTransactionsDict[transaction] = lastFrom
    lastFrom += 1
    return intFromTransactionsDict[transaction]
    
intCompTransactionsDict = {}
lastComp = 1

def getIntCompTransaction(transaction):
    global intCompTransactionsDict
    global lastComp
    if transaction in intCompTransactionsDict:
        return intCompTransactionsDict[transaction]
    intCompTransactionsDict[transaction] = lastComp
    lastComp += 1
    return intCompTransactionsDict[transaction]

intToTransactionsDict = {}
lastTo = 1

def getIntToTransaction(transaction):
    global intToTransactionsDict
    global lastTo
    if transaction in intToTransactionsDict:
        return intToTransactionsDict[transaction]
    intToTransactionsDict[transaction] = lastTo
    lastTo += 1
    return intToTransactionsDict[transaction]