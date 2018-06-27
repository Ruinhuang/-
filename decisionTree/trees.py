from math import log
import operator
# coding: utf-8

def calcShannonEnt (dataSet):
    '''
    用来计算给定数据集的香农熵
    return:香农熵
    '''
    # 求list的长度，表示计算参与训练的数据量
    numEntries = len(dataSet)
    # 创建一个字典，键值是最后一列的数值,计算label出现的次数
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        # 为所有可能的分类创建字典，如果当前键值不存在，则扩展字典并将当前键值加入字典
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        # 每个键值记录了当前类别出现的次数
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    # 使用所有类标签发生频率计算类别出现概率
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        # 以2为底求对数
        # 使用概率计算香农熵，统计所有标签发生的次数
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def createDataSet():
    '''
    测试数据
    书page35：海洋生物数据
    :return:
    '''
    dateSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dateSet, labels

def splitDataSet(dataSet, axis, value):
    '''
    按照给定的特征划分数据集
    就是依据axis列进行分类，如果index列的数据等于 value的时候，就要将 axis 划分到我们创建的新的数据集中
    :param dataSet:待划分的数据集
    :param axis:划分数据集的特征
    :param value:需要返回的特征的值
    :return:
    '''
    # 创建一个新的列表对象
    retDataSet = []
    for featVec in dataSet:
        # 判断index列的值是否为value
        if featVec[axis] == value:
            # [:axis]表示前axis列
            reducedFeatVec = featVec[:axis]
            # 使用append的时候，是将object看作一个对象，整体打包添加到music_media对象中。
            # 使用extend的时候，是将sequence看作一个序列，将这个序列和music_media序列合并，并放在其后面
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureTopSplit(dataSet):
    '''
    遍历整个数据集
    循环计算香农熵和splitDataSet函数
    找到最好的特征划分方式
    在函数中调用的数据需要满足两个要求：
        1.数据必须是一种由列表元素组成的列表，而且所有的列表元素都要有相同的数据长度
        2.数据的最后一列或者每个实例的最后一个元素是当前实例的类别标签
    '''
    # 求第一行有多少列的 Feature, 最后一列是label列嘛
    numFeatures = len(dataSet[0]) - 1
    # 计算整个数据集的原始香农熵
    baseEntropy = calcShannonEnt(dataSet)
    # 最优信息增益值
    bestInfoGain = 0.0
    # 最优的feature编号
    bestFeature = -1
    # 遍历数据集中的所有特征
    for i in range(numFeatures):
        # 列表推导式快速生成列表，将数据集中所有第i个特征值或者所有可能存在的值写入featList中
        # 获取对应的feature下的所有数据
        featList = [example[i] for example in dataSet]
        # set与列表类似，消除了列表中的相同的值
        uniqueVals = set(featList)
        # 创建一个临时的信息熵
        newEntropy = 0.0
        # 遍历某一列的value集合，计算出每种划分方式的香农熵
        # 遍历当前特征值中所有的唯一属性
        for value in uniqueVals:
            # 对每个唯一属性划分一次数据集
            subDataSet = splitDataSet(dataSet, i, value)
            # 计算概率
            prob = len(subDataSet)/float(len(dataSet))
            # 计算信息熵
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 选取最好的信息增益
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    '''

    :param classList:
    :return: 返回出现次数最多的分类名称
    '''
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    '''
    sort用法：
    sorted(iterable[, cmp[, key[, reverse]]])
    iterable -- 可迭代对象。
    cmp -- 比较的函数，这个具有两个参数，参数的值都是从可迭代对象中取出，此函数必须遵守的规则为，大于则返回1，小于则返回-1，等于则返回0。
    key -- 主要是用来进行比较的元素，只有一个参数，具体的函数的参数就是取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序。
    reverse -- 排序规则，reverse = True 降序 ， reverse = False 升序（默认）。
    '''
    # python3用items代替iteritems,用来进行遍历
    # 字典(Dictionary) items() 函数以列表返回可遍历的(键, 值) 元组数组
    # 要通过classCount.items的第1个域排序，用key=operator.itemgetter(1)对键值进行排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    '''
    创建树的函数代码
    :param dataSet:数据集
    :param labels:标签列表
    :return:
    '''
    # 创建列表，包含了数据集的所有类标签
    classList = [example[-1] for example in dataSet]
    # Python count() 方法用于统计字符串里某个字符出现的次数
    # 递归函数的第一个停止条件是所有的类标签完全相同，则直接返回该类标签
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 第二个停止条件是使用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分组，则用majorityCnt函数返回出现次数最多的类别
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 创建树
    # 选择最优的列，得到最优列对应的label含义
    bestFeat = chooseBestFeatureTopSplit(dataSet)
    # 获取label的名称
    bestFeatLabel = labels[bestFeat]
    # 初始化myTree
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])
    # 取出最优列，然后它的branch做分类
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # 求出剩余的标签label
        subLabels = labels[:]
        # 遍历当前选择特征包含的所有属性值，在每个数据集划分上递归调用函数createTree()
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)

    return myTree

def classify(inputTree,featLabels,testVec):
    '''
    测试算法，使用决策树执行分类
    :param inputTree:决策树模型
    :param featLabels:标签对应的名称
    :param testVec:测试输入的数据
    :return:classLabel 分类的结果值，需要映射label才能知道名称
    '''
    # 获取tree的根节点对于的key值
    firstStr = list(inputTree.keys())[0]
    # 通过key得到根节点对应的value
    secondDict = inputTree[firstStr]
    # 使用index方法查找当前列表中第一个匹配firstStr变量的元素
    # 判断根节点名称获取根节点在label中的先后顺序，这样就知道输入的testVec怎么开始对照树来做分类
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

def storeTree(inputTree,filename):
    '''
    使用pickle模块存储决策树
    :param inputTree:
    :param filename:
    :return:
    '''
    import pickle
    fw = open(filename, 'wb')
    # 将一个对象转储为一个字符串
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):
    '''
    pickle模块序列化
    :param filename:
    :return:
    '''
    import pickle
    fr = open(filename, 'rb')
    # 从字节流中恢复一个对象
    return pickle.load(fr)
