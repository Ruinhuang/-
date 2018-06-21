# 机器学习实战（1）--- KNN

## 环境

- Windows/Pycharm/Python 3.6

## 原理

这是机器学习实战这本书的第二章内容，也是本书研究的第一个算法，K近邻算法网上也有很多的教程，主要原理如下：

1. 计算已知类别数据集中的点与当前点之间的距离；
2. 按照距离递增次序排序；
3. 选取与当前点距离最小的k个点；
4. 确定k个点所在类别的出现频率；
5. 返回前k个点出现频率最高的类别作为当前点的预测分类

## 代码

### KNN算法代码

```python
#KNN算法
def classify0 (inX, dataSet, labels, k):
    '''
    inX:输入的测试数据
    dataSet: 训练数据集
    labels:标签
    k:
    return:所属类别
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
```

- 此处距离公式为欧式距离公式

  
  $$
  distance = \sqrt{(xA_0 - xB_0)^2 + (xA_1 - xB_1)^2 }
  $$

- 除了欧式距离还可以换成别的距离计算方式

### 1. 使用KNN算法改进约会网站匹配效果

- 简单的将约会网站的人群分为三类：A.不喜欢的人 B.魅力一般的人 C.极具魅力的人

- 收集的数据在datingTestSet2.txt中，每个样本数据占一行，共有1000行；样本数据主要包含一下三种特征：

  1. 每年飞行里程数
  2. 玩视频游戏消耗的时间百分比
  3. 每周消耗的冰淇淋数

  ![](https://www.z4a.net/images/2018/06/20/1bc8c2774da2d6936.png)

#### 将文本记录转化为NumPy的解析程序

```python
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
```

#### 归一化

- 为了使三个特征值的权重相同需要对数据进行归一化处理

  ```python
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
  ```

  

- tile函数可以将变量内容复制成输入矩阵相同大小的矩阵

- 结果使得所有特征值均在0~1之间

#### 测试算法

- 数据的90%作为训练样本

- 10%测试样本评估KNN的测试分类器

  ```python
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
  ```

  

- 结果为：![](https://www.z4a.net/images/2018/06/20/26cb22b3af6e83fbb.png)

- 错误率为5%，改变hoRatio及k的值输出结果会有不同

  ​                 

#### 使用算法

  - 在约会网站上找到某个人并输入他的信息，程序会预测出对对方的喜欢程度

    ```python
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
        #预测
        inArr = array([ffMiles, percentTats, iceCream])
        classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
        print("You will probably like this person: ", resultList[classifierResult - 1])
    ```

- 结果：![](https://www.z4a.net/images/2018/06/20/3bb32321704540910.png)



### 2. 手写识别

- 识别数字1~9

- 宽高为32*32像素的黑白图像

#### 将图像转换为测试向量

- 文件夹trainingDigits中包含了2000个例子，每个数字大约有200个样本，用来训练分类器

- 文件夹testDigits中包含了大约900个测试数据，用来测试分类器效果

- 首先需要将32x32的二进制图像矩阵转化为1x1024的向量

  ```python
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
  ```

  

#### 测试算法

- from os import listdir：可以列出给定目录的文件名

```python
#测试KNN算法
def handwritingClassTest():
    '''
    每个文件的文件名前缀为数字几的标签，用文件名获得label
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
```

- 结果：![](https://www.z4a.net/images/2018/06/20/4.png)
- 错误率略高，可以调整k，改变训练样本数目等方式可以调整算法