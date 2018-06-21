# coding: utf-8

from numpy import *
import operator
#列出给定目录的文件名
from os import listdir

#创建数据集和标签
def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

#KNN算法
def classify0 (inX, dataSet, labels, k):
    '''

    inX:输入的测试数据
    dataSet: 训练数据集
    labels:标签
    k:
    :return:所属 类别
    '''
    # 距离计算
    dataSetSize = dataSet.shape[0]

    #tile：测试数据与训练样本求差
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    #平方
    sqDiffMat = diffMat**2
    #求和，矩阵每一行相加
    sqDistances = sqDiffMat.sum(axis=1)
    #开方
    distances = sqDistances**0.5

    # 将距离排序：从小到大
    sortedDistIndicies = distances.argsort()

    #选择距离最小的k个点
    classCount = {}
    for i in range(k):
        #找到测试样本类型
        voteIlabel = labels[sortedDistIndicies[i]]
        #对找到的该分类加一
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    #排序并返回最多的那个类型
    #python3.5以上iteritems变为items
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

#数据预处理,处理数据输入格式
def file2matrix(filename):
    '''
    输入为文件名字符串
    输出为训练样本矩阵和类标签向量
    '''
    fr = open(filename)
    arrayOLines = fr.readlines()
    #得到文件行数
    numbeerOfLines = len(arrayOLines)
    #创建返回的numpy矩阵
    returnMat = zeros((numbeerOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        #strip删除空格
        line = line.strip()
        listFromLine = line.split('\t')
        #选取前三个元素将其储存到特征矩阵中
        returnMat[index, :] = listFromLine[0:3]
        #索引-1表示列表中最后一列元素
        classLabelVector.append(int(listFromLine[-1]))
        index += 1

    return returnMat, classLabelVector

#归一化特征值
def autoNorm(dataSet):
    '''
    用当前值减去最小值再除以最大值与最小值的差即可归一化
    '''
    #参数0可以使函数从列中选取最小值而不是当前行的最小值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

#测试方法
def datingClassTest():
    hoRatio = 0.10
    #加载数据
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    #归一化
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    #测试数据
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is : %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]) : errorCount += 1.0
    print("The total error rate is: %f" % (errorCount/float(numTestVecs)))

#预测函数
def classifyPerson():
    '''
    找到一个人根据给出的条件预测对对方的喜欢程度
    三类人：不喜欢，魅力一般，极具魅力
    '''

    resultList = ['not at all', 'in some doses', 'in large doses']
    #玩视频游戏消耗的百分比
    percentTats = float(input("Percentage of time spent playing video games?"))
    #每年飞行的距离数
    ffMiles = float(input("frequent flier miles earned per year?"))
    #每周消费的冰淇淋数
    iceCream = float(input("liters of ice cream consumed per year?"))
    #训练数据
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    #归一化
    normMat, ranges, minVals = autoNorm(datingDataMat)

    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print("You will probably like this person: ", resultList[classifierResult - 1])


##########################################################################
##########################2.3KNN手写识别###################################
##########################################################################

#将图像转化为向量

def img2vector(filename):
    '''
    将创建1x1024的Numpy数组
    然后打开指定文件，循环读出文件前32行，并将每行的头32个字符值储存在Numpy数组中
    最后返回数组
    '''
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect

#测试KNN算法
def handwritingClassTest():
    '''
    每个文件的文件名前缀为数字几的标签
    '''
    hwLabels = []
    #获取文件目录
    trainingFileList = listdir('trainingDigits')
    #得到目录中的文件数量
    m = len(trainingFileList)
    #创建一个m行的1024矩阵，每行数据储存一个图像
    trainingMat = zeros((m, 1024))
    for i in range(m):
        # 获取到文件名的所代表的数字
        # eg: 9_45.txt代表数字9的第45个实例
        fileNameStr = trainingFileList[i]
        #去掉txt后缀
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        #调用img2vector将图像转化为向量，得到trainingSet
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    #对testDigits目录中的文件执行相同的操作
    testFileList = listdir('testDigits')
    errorcount = 0.0
    mTets = len(testFileList)
    for i in range(mTets):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' %fileNameStr)
        #调用KNN算法
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("The classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorcount += 1.0
    print("\nThe total number of error is: %d" % errorcount)
    print("\nThe total error rate is: %f" % (errorcount/float(mTets)))