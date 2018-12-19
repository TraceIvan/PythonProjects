def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def createC1(dataSet):
    C1 = []#元素个数为1的项集（非频繁项集，因为还没有同最小支持度比较）
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))


def scanD(D, Ck, minSupport):
    """D为全部数据集，Ck为大小为k（包含k个元素）的候选项集，minSupport为设定的最小支持度"""
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if can not in ssCnt:
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    #retList为在Ck中找出的频繁项集（支持度大于minSupport的），supportData记录各频繁项集的支持度。
    return retList, supportData

'''
当集合中项的个数大于0时：
    构建一个由k个项组成的候选项集的列表（k从1开始）
    计算候选项集的支持度，删除非频繁项集
    构建由k+1项组成的候选项集的列表
'''
def aprioriGen(Lk, k):
    '''通过频繁项集列表Lk和项集个数k生成候选项集Ck+1'''
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            L1 = list(Lk[i])[:k - 2]#取列表的前k-1个元素
            L2 = list(Lk[j])[:k - 2]
            L1.sort()
            L2.sort()
            if L1 == L2:  #前k-1项相同时
                retList.append(Lk[i] | Lk[j])  # 两项合并
    return retList

'''主函数'''
def apriori(dataSet, minSupport=0.5):
    '''Ck表示项数为k的候选项集'''
    C1 = createC1(dataSet)#C1通过createC1()函数生成
    D = list(map(set,dataSet))
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k - 2]) > 0):
        Ck = aprioriGen(L[k - 2], k)
        Lk, supK = scanD(D, Ck, minSupport)  #Lk表示项数为k的频繁项集,supK为其支持度
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData#所有的频繁项集及其支持度

#关联规则生成函数
def generateRules(L, supportData, minConf=0.7):
    '''频繁项集列表L、包含那些频繁项集支持数据的字典supportData、最小可信度阈值minConf'''
    bigRuleList = []
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList

#对规则进行评估（计算支持度）
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        if conf >= minConf:
            print(freqSet - conseq, '-->', conseq, 'conf:', conf)
            brl.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

#生成候选规则集合
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    while (len(freqSet) > m):  # 判断长度 > m，这时即可求H的可信度
        H = calcConf(freqSet, H, supportData, brl, minConf)
        if (len(H) > 1):  # 判断求完可信度后是否还有可信度大于阈值的项用来生成下一层H
            H = aprioriGen(H, m + 1)
            m += 1
        else:  # 不能继续生成下一层候选关联规则，提前退出循环
            break


def pntRules(ruleList, itemMeaning):
    for ruleTup in ruleList:
        for item in ruleTup[0]:
            print(itemMeaning[item])
        print("           -------->")
        for item in ruleTup[1]:
            print(itemMeaning[item])
        print("confidence: %f" % ruleTup[2])
        print()  # print a blank line


from time import sleep
from votesmart import votesmart

votesmart.apikey = 'get your api key first'


def getActionIds():
    actionIdList = []
    billTitleList = []
    fr = open('recent20bills.txt')
    for line in fr.readlines():
        billNum = int(line.split('\t')[0])
        try:
            billDetail = votesmart.votes.getBill(billNum)  # api call
            for action in billDetail.actions:
                if action.level == 'House' and \
                        (action.stage == 'Passage' or action.stage == 'Amendment Vote'):
                    actionId = int(action.actionId)
                    print('bill: %d has actionId: %d' % (billNum, actionId))
                    actionIdList.append(actionId)
                    billTitleList.append(line.strip().split('\t')[1])
        except:
            print("problem getting bill %d" % billNum)
        sleep(1)  # delay to be polite
    return actionIdList, billTitleList


def getTransList(actionIdList, billTitleList):  # this will return a list of lists containing ints
    itemMeaning = ['Republican', 'Democratic']  # list of what each item stands for
    for billTitle in billTitleList:  # fill up itemMeaning list
        itemMeaning.append('%s -- Nay' % billTitle)
        itemMeaning.append('%s -- Yea' % billTitle)
    transDict = {}  # list of items in each transaction (politician)
    voteCount = 2
    for actionId in actionIdList:
        sleep(3)
        print('getting votes for actionId: %d' % actionId)
        try:
            voteList = votesmart.votes.getBillActionVotes(actionId)
            for vote in voteList:
                if not transDict.has_key(vote.candidateName):
                    transDict[vote.candidateName] = []
                    if vote.officeParties == 'Democratic':
                        transDict[vote.candidateName].append(1)
                    elif vote.officeParties == 'Republican':
                        transDict[vote.candidateName].append(0)
                if vote.action == 'Nay':
                    transDict[vote.candidateName].append(voteCount)
                elif vote.action == 'Yea':
                    transDict[vote.candidateName].append(voteCount + 1)
        except:
            print("problem getting actionId: %d" % actionId)
        voteCount += 2
    return transDict, itemMeaning
