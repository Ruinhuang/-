#  机器学习实战（5）--- 支持向量机（SVM）

## 环境

- Windows/Pycharm/Python 3.6

## 原理

### SVM

- SVM原理有点难以理解，此处引用知乎大神  [@简之]: https://www.zhihu.com/question/21094489/answer/86273196的回答

  > 魔鬼在桌子上似乎有规律放了两种颜色的球，说：“你用一根棍分开它们？要求：尽量在放更多球之后，仍然适用。
  > ![](https://www.z4a.net/images/2018/07/05/155b01de106d392b9.png)
  >
  > **SVM就是试图把棍放在最佳位置，好让在棍的两边有尽可能大的间隙。**
  >
  > ![](https://www.z4a.net/images/2018/07/05/26e6f859f5e42fb30.png)
  >
  > 然后，在SVM 工具箱中有另一个更加重要的 **trick**。 魔鬼看到大侠已经学会了一个trick，于是魔鬼给了大侠一个新的挑战
  >
  > ![](https://www.z4a.net/images/2018/07/05/3ecc78cfbc2097043.png)
  >
  > 现在，大侠没有棍可以很好帮他分开两种球了，现在怎么办呢？当然像所有武侠片中一样大侠桌子一拍，球飞到空中。然后，凭借大侠的轻功，大侠抓起一张纸，插到了两种球的中间。
  >
  > ![](https://www.z4a.net/images/2018/07/05/49129a71cb8911960.png)
  >
  > 现在，从魔鬼的角度看这些球，这些球看起来像是被一条曲线分开了
  >
  > ![](https://www.z4a.net/images/2018/07/05/58610c82c9eaf0a10.png)
  >
  > 再之后，无聊的大人们，把这些球叫做 **「data」**，把棍子 叫做 **「classifier」**, 最大间隙trick 叫做**「optimization」**， 拍桌子叫做**「kernelling」**, 那张纸叫做**「hyperplane」**
  >
  > 视频链接 https://www.youtube.com/watch?v=3liCbRZPrZA

- 关于机器学习的数学推导：https://www.zhihu.com/question/21094489/answer/117246987

### SMO高效优化算法

- SVM有很多种实现，最流行的一种实现是： 序列最小优化(Sequential Minimal Optimization, SMO)算法

- SMO用途：用于训练 SVM

- SMO目标：求出一系列 alpha 和 b,一旦求出 alpha，就很容易计算出权重向量 w 并得到分隔超平面

- SMO思想：是将大优化问题分解为多个小优化问题来求解的

- SMO原理：每次循环选择两个 alpha 进行优化处理，一旦找出一对合适的 alpha，那么就增大一个同时减少一个

  - 这里指的合适必须要符合一定的条件 

    1. 这两个 alpha 必须要在间隔边界之外

    2. 这两个 alpha 还没有进行过区间化处理或者不在边界上
  - 之所以要同时改变2个 alpha；原因是我们有一个约束条件： $ \sum_{i=1}^{m} a_i·label_i=0$；如果只是修改一个 alpha，很可能导致约束条件失效

- SMO的伪代码

  ```
  创建一个 alpha 向量并将其初始化为0向量
  当迭代次数小于最大迭代次数时(外循环) 
  	对数据集中的每个数据向量(内循环)：
  		如果该数据向量可以被优化：
  			随机选择另外一个数据向量
  			同时优化这两个向量
  	如果两个向量都不能被优化，退出内循环
  ```

- SVM 算法特点

	```
	优点：泛化（由具体的、个别的扩大为一般的，就是说：模型训练完后的新样本）错误率低，	计算开销不大，结果易理解。
	缺点：对参数调节和核函数的选择敏感，原始分类器不加修改仅适合于处理二分类问题。
	使用数据类型：数值型和标称型数据。
	```

## 代码

### 对小规模数据点进行分类

#### 收集数据

- 数据格式：

```
3.542485	1.977398	-1
3.018896	2.556416	-1
7.551510	-1.580030	1
2.114999	-0.004466	-1
8.127113	1.274372	1
7.108772	-0.986906	1
8.610639	2.046708	1
2.326297	0.265213	-1
3.634009	1.730537	-1
0.341367	-0.894998	-1
```

#### 准备数据

```python
def loadDataSet(fileName):

    """loadDataSet（对文件进行逐行解析，从而得到第行的类标签和整个数据矩阵）
    Args:
        fileName 文件名
    Returns:
        dataMat  数据矩阵
        labelMat 类标签
    """
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

def selectJrand(i, m):
    '''
    只要函数值不等于输入值i，函数就会随机选择一个整数
    :param i:第一个alpha的下标
    :param m:所有alpha的数目
    :return:返回一个不为i的随机数，在0~m之间
    '''
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j

def clipAlpha(aj, H, L):
    '''
    clipAlpha(调整aj的值，使aj处于 L<=aj<=H)
    :param aj:目标值
    :param H:最大值
    :param L: 最小值
    :return: aj  目标值
    '''
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj
```

- 结果测试：

![](https://www.z4a.net/images/2018/07/06/684034fdc1f46493f.png)

#### 简化版SMO算法

```python
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    '''

    :param dataMatIn:数据集
    :param classLabels: 类别标签
    :param C: 松弛变量(常量值)，允许有些数据点可以处于分隔面的错误一侧。
            控制最大化间隔和保证大部分的函数间隔小于1.0这两个目标的权重。
            可以通过调节该参数达到不同的结果。
    :param toler: 容错率（是指在某个体系中能减小一些因素或选择对某个系统产生不稳定的概率。）
    :param maxIter: 退出前最大的循环次数
    :return:
    b 模型的常量值
    alphas  拉格朗日乘子
    '''
    dataMatrix = mat(dataMatIn)
    # 转置
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)

    # 初始化 b和alphas(alpha有点类似权重值)
    b = 0
    alphas = mat(zeros((m, 1)))

    # 没有任何alpha改变的情况下遍历数据的次数
    iter = 0
    while (iter < maxIter):
        # 记录alpha是否已经进行优化，每次循环时设为0，然后再对整个集合顺序遍历
        alphaPairsChanged = 0
        for i in range(m):
            # 预测的类别 y = w^Tx[i]+b; 其中 w = Σ(1~n) a[n]*lable[n]*x[n]
            fXi = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T) + b)
            # 预测结果与真实结果比对，计算误差Ei
            Ei = fXi - float(labelMat[i])
            # 约束条件 (KKT条件是解决最优化问题的时用到的一种方法。我们这里提到的最优化问题通常是指对于给定的某一函数，求其在指定作用域上的全局最小值)
            # 0<=alphas[i]<=C，但由于0和C是边界值，我们无法进行优化，因为需要增加一个alphas和降低一个alphas。
            # 表示发生错误的概率：labelMat[i]*Ei 如果超出了 toler， 才需要优化。至于正负号，我们考虑绝对值就对了。
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                # 如果满足优化的条件，我们就随机选取非i的一个点，进行优化比较
                j = selectJrand(i, m)
                # 预测j的结果
                fXj = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                # L和H用于将alphas[j]调整到0-C之间。如果L==H，就不做任何改变，直接执行continue语句
                # labelMat[i] != labelMat[j] 表示异侧，就相减，否则是同侧，就相加
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                # 如果相同，不优化
                if L == H:
                    print("L == H")
                    continue

                # eta是alphas[j]的最优修改量，如果eta==0，需要退出for循环的当前迭代过程
                # 参考《统计学习方法》李航-P125~P128<序列最小最优化算法>
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * dataMatrix[i, :].T - dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0:
                    print("eta>=0")
                    continue

                # 计算出一个新的alphas[j]值
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                # 并使用辅助函数，以及L和H对其进行调整
                alphas[j] = clipAlpha(alphas[j], H, L)
                # 检查alpha[j]是否只是轻微的改变，如果是的话，就退出for循环。
                # abs() 函数返回数字的绝对值。
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print("j not moving enough")
                    continue
                # 然后alphas[i]和alphas[j]同样进行改变，虽然改变的大小一样，但是改变的方向正好相反
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                # 在对alpha[i], alpha[j] 进行优化之后，给这两个alpha值设置一个常数b。
                # w= Σ[1~n] ai*yi*xi => b = yj- Σ[1~n] ai*yi(xi*xj)
                # 所以：  b1 - b = (y1-y) - Σ[1~n] yi*(a1-a)*(xi*x1)
                # 为什么减2遍？ 因为是 减去Σ[1~n]，正好2个变量i和j，所以减2遍
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif(0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print("iter: %d i: %d, pairs changed %d" % (iter, i, alphaPairsChanged))
        # 在for循环外，检查alpha值是否做了更新，如果在更新则将iter设为0后继续运行程序
        # 知道更新完毕后，iter次循环无变化，才推出循环
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print("iteration number: %d" % iter)
    return b, alphas
```

- 结果：

![](https://www.z4a.net/images/2018/07/06/782ef9d0bc0971540.png)

#### 完整版的SMO算法

- 由于简化版的SMO算法在大数据集上运行速度比较慢，所以对其进行优化
- 和简化版唯一不同在于选择alpha方式

##### 支持函数

```python
class optStruct:
    '''
    建立的数据结构来保存所有的重要值
    '''
    def __init__(self, dataMatIn, classLabels, C, toler):
        '''

        :param dataMatIn:数据集
        :param classLabels:类别标签
        :param C:松弛变量(常量值)，允许有些数据点可以处于分隔面的错误一侧。
                控制最大化间隔和保证大部分的函数间隔小于1.0这两个目标的权重。
                可以通过调节该参数达到不同的结果。
        :param toler:容错率
        :param kTup:包含核函数信息的元组
        '''
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))


def calcEk(oS, k):
    '''
    calcEk（求 Ek误差：预测值-真实值的差）
    该过程在完整版的SMO算法中陪出现次数较多，因此将其单独作为一个方法
    :param oS:optStruct对象
    :param k:具体的某一行


    :return: Ek  预测结果与真实结果比对，计算误差Ek
    '''
    fXk = multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k].T) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def selectJ(i, oS, Ei):
    '''
    selectJ（返回最优的j和Ej）
    内循环的启发式方法。
    选择第二个(内循环)alpha的alpha值
    这里的目标是选择合适的第二个alpha值以保证每次优化中采用最大步长。
    该函数的误差与第一个alpha值Ei和下标i有关。
    :param i:具体的第i一行
    :param oS:optStruct对象
    :param Ei:预测结果与真实结果比对，计算误差Ei

    :return:j  随机选出的第j一行
           Ej 预测结果与真实结果比对，计算误差Ej
    '''
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    # 首先将输入值Ei在缓存中设置成为有效的。这里的有效意味着它已经计算好了。
    oS.eCache[i] = [1, Ei]
    # print('oS.eCache[%s]=%s' % (i, oS.eCache[i]))
    # print('oS.eCache[:, 0].A=%s' % oS.eCache[:, 0].A.T)
    # """
    # # 返回非0的：行列值
    # nonzero(oS.eCache[:, 0].A)= (
    #     行： array([ 0,  2,  4,  5,  8, 10, 17, 18, 20, 21, 23, 25, 26, 29, 30, 39, 46,52, 54, 55, 62, 69, 70, 76, 79, 82, 94, 97]),
    #     列： array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0])
    # )
    # """
    # print('nonzero(oS.eCache[:, 0].A)=', nonzero(oS.eCache[:, 0].A))
    # # 取行的list
    # print('nonzero(oS.eCache[:, 0].A)[0]=', nonzero(oS.eCache[:, 0].A)[0])
    # 非零E值的行的list列表，所对应的alpha值
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]
    if (len(validEcacheList)) > 1:
        # 在所有的值上进行循环，并选择其中使得改变最大的那个值
        for k in validEcacheList:
            if k == i:
                continue

            # 求 Ek误差：预测值-真实值的差
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE:
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        # 如果是第一次循环，则随机选择一个alpha值
        j = selectJrand(i, oS.m)
    return j, Ej

def updateEk(oS, k):
    '''
    updateEk（计算误差值并存入缓存中）
     在对alpha值进行优化之后会用到这个值
    :param oS:optStruct对象
    :param k:某一列的行号
    :return:
    '''
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]
```

##### 优化例程

```python
def innerL(i, oS):
    '''
     内循环代码
    :param i:具体的某一行
    :param oS:optStruct对象

    :return: 0   找不到最优的值
            1   找到了最优的值，并且oS.Cache到缓存中
    '''
    # 求 Ek误差：预测值-真实值的差
    Ei = calcEk(oS, i)
    # 约束条件 (KKT条件是解决最优化问题的时用到的一种方法。我们这里提到的最优化问题通常是指对于给定的某一函数，求其在指定作用域上的全局最小值)
    # 0<=alphas[i]<=C，但由于0和C是边界值，我们无法进行优化，因为需要增加一个alphas和降低一个alphas。
    # 表示发生错误的概率：labelMat[i]*Ei 如果超出了 toler， 才需要优化。至于正负号，我们考虑绝对值就对了
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        # 选择最大的误差对应的j进行优化。效果更明显
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()

        # L和H用于将alphas[j]调整到0-C之间。如果L==H，就不做任何改变，直接return 0
        if oS.labelMat[i] != oS.labelMat[j]:
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L == H")
            return 0

        # eta是alphas[j]的最优修改量，如果eta==0，需要退出for循环的当前迭代过程
        # 参考《统计学习方法》李航-P125~P128<序列最小最优化算法>
        eta = oS.X[i] - oS.X[j]
        eta = -eta * eta.T
        if eta >= 0:
            print("eta >= 0")
            return 0

        # 计算出一个新的alphas[j]值
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        # 并使用辅助函数，以及L和H对其进行调整
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        # 更新误差缓存
        updateEk(oS, j)

        # 检查alpha[j]是否只是轻微的改变，如果是的话，就退出for循环
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print(" j not moving enough")
            return 0

        # 然后alphas[i]和alphas[j]同样进行改变，虽然改变的大小一样，但是改变的方向正好相反
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        # 更新误差缓存
        updateEk(oS, i)

        # 在对alpha[i], alpha[j] 进行优化之后，给这两个alpha值设置一个常数b。
        # w= Σ[1~n] ai*yi*xi => b = yj Σ[1~n] ai*yi(xi*xj)
        # 所以：  b1 - b = (y1-y) - Σ[1~n] yi*(a1-a)*(xi*x1)
        # 为什么减2遍？ 因为是 减去Σ[1~n]，正好2个变量i和j，所以减2遍
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i] * oS.X[i].T - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[i] * oS.X[j].T
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i] * oS.X[j].T - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[j] * oS.X[j].T
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2
        return 1
    else:
        return 0
```

##### 外循环代码

```python
def smoP(dataMatIn, classLabels, C, totler, maxIter):
    '''
    完整SMO算法外循环，与smoSimple有些类似，但这里的循环退出条件更多一些
    :param dataMatIn:数据集
    :param classLabels:类别标签
    :param C:松弛变量(常量值)，允许有些数据点可以处于分隔面的错误一侧。
            控制最大化间隔和保证大部分的函数间隔小于1.0这两个目标的权重。
            可以通过调节该参数达到不同的结果。
    :param totler:容错率
    :param maxIter:退出前最大的循环次数
    :return:
     b       模型的常量值
    alphas  拉格朗日乘子
    '''
    # 创建一个 optStruct 对象
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, totler)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0

    # 循环遍历：循环maxIter次 并且 （alphaPairsChanged存在可以改变 or 所有行遍历一遍）
    # 循环迭代结束 或者 循环遍历所有alpha后，alphaPairs还是没变化
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        # ----------- 第一种写法 start -------------------------
        #  当entireSet=true or 非边界alpha对没有了；就开始寻找 alpha对，然后决定是否要进行else
        if entireSet:
            # 在数据集上遍历所有可能的alpha
            for i in range(oS.m):
                # 是否存在alpha对，存在就+1
                alphaPairsChanged += innerL(i, oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        # 对已存在 alpha对，选出非边界的alpha值，进行优化
        else:
            # 遍历所有的非边界alpha值，也就是不在边界0或C上的值。
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        # ----------- 第一种写法 end -------------------------

        # ----------- 第二种方法 start -------------------------
        # if entireSet:																				#遍历整个数据集
        # 	alphaPairsChanged += sum(innerL(i, oS) for i in range(oS.m))
        # else: 																						#遍历非边界值
        # 	nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]						#遍历不在边界0和C的alpha
        # 	alphaPairsChanged += sum(innerL(i, oS) for i in nonBoundIs)
        # iter += 1
        # ----------- 第二种方法 end -------------------------

        # 如果找到alpha对，就优化非边界alpha值，否则，就重新进行寻找，如果寻找一遍 遍历所有的行还是没找到，就退出循环
        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:
            entireSet = True
        print("iteration number: %d" % iter)

    return oS.b, oS.alphas

def calcWs(alphas, dataArr, classLabels):
    '''
    基于alpha计算w值
    :param alphas:拉格朗日乘子
    :param dataArr:feature数据集
    :param classLabels:目标变量数据集
    :return:wc  回归系数
    '''
    X = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i].T)
    return w
```

- ![结果](https://www.z4a.net/images/2018/07/06/87afe0386f0acdbfe.png)

### 核函数

- 对于非线性的情况也一样，此时需要用到一种叫核函数(kernel)的工具将数据转化为分类器易于理解的形式
- 利用核函数将数据映射到高维空间
- 使用核函数：可以将数据从某个特征空间到另一个特征空间的映射（通常情况下：这种映射会将低维特征空间映射到高维空间）
- 经过空间转换后：低维需要解决的非线性问题，就变成了高维需要解决的线性问题
- SVM 优化特别好的地方，在于所有的运算都可以写成内积(inner product: 是指2个向量相乘，得到单个标量 或者 数值)；内积替换成核函数的方式被称为核技巧(kernel trick)
- 核函数并不仅仅应用于支持向量机，很多其他的机器学习算法也都用到核函数。最流行的核函数：径向基函数(radial basis function)

##### 核转换函数

```python
class optStruct:
    '''
    建立的数据结构来保存所有的重要值
    '''
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        '''

        :param dataMatIn:数据集
        :param classLabels:类别标签
        :param C:松弛变量(常量值)，允许有些数据点可以处于分隔面的错误一侧。
                控制最大化间隔和保证大部分的函数间隔小于1.0这两个目标的权重。
                可以通过调节该参数达到不同的结果。
        :param toler:容错率
        :param kTup:包含核函数信息的元组
        '''
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        # 数据的行数
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        # 误差缓存，第一列给出的是eCache是否有效的标志位，第二列给出的是实际的E值。
        self.eCache = mat(zeros((self.m, 2)))
        # m行m列的矩阵
        self.K = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i], kTup)

def kernelTrans(X, A, kTup):
    '''
    核转换函数
    :param X:dataMatIn数据集
    :param A:dataMatIn数据集的第i行的数据
    :param kTup:核函数的信息
    :return:
    '''

    m, n = shape(X)
    K = mat(zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        # 径向基函数的高斯版本
        K = exp(K / (-1 * kTup[1] ** 2))
    else:
        raise NameError('Houston We Have A Problem -- That Kernel is Not Recognized')
    return K
```

##### 利用核函数进行分类的径向基测试函数

```python
def testRbf(k1 = 1.3):
    dataArr, labelArr = loadDataSet('testSetRBF.txt')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print("There are %d support vectors" % shape(sVs)[0])
    m, n = shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], ('rbf', k1))
        # 和这个svm-simple类似： fXi = float(multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[i, :].T)) + b
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print("The training error rate is: %f" % (float(errorCount) / m))

    dataArr, labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m, n = shape(dataMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print("The test error rate is: %f" % (float(errorCount) / m))

```

- 结果![](https://www.z4a.net/images/2018/07/06/94faf302c77bd5f74.png)

### 手写识别+核函数

```python
def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

def loadImages(dirName):
    from os import listdir
    hwLabel = []
    print(dirName)
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9:
            hwLabel.append(-1)
        else:
            hwLabel.append(1)
        trainingMat[i, :] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabel

def testDigits(kTup=('rbf', 10)):
    # 1. 导入训练数据
    dataArr, labelArr = loadImages('trainingDigits')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    m, n = shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], kTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        # sign函数就是大于0的返回1.0;小于0的返回-1.0;等于0的返回0.0
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print('The training error rate is: %f' % (float(errorCount) / m))
    # 2.导入测试数据
    dataArr, labelArr = loadImages('testDigits')
    errorCount = 0
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m, n = shape(dataMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], kTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print("The test error rate is: %f" % (float(errorCount) / m))

```

- 结果![](https://www.z4a.net/images/2018/07/06/x.png)