#  机器学习实战（4） --- Logistics回归

## 环境

- Windows/Pycharm/Python 3.6

## 原理

### 逻辑回归

- 是用来做分类任务的

- 主要思想是：根据现有的数据对**分类边界线**建立回归公式，以此进行分类

- 优点：计算代价不高，易于理解和实现；缺点：容易欠拟合，分类精度不高

- 适用数据类型：数值型和标称性

- 想要构建的分类函数：接受所有的输入然后预测出类别

### sigmoid函数



$$
  \sigma(z)=\frac{1}{1+e^{-z}}
$$


 ![图像](https://www.z4a.net/images/2018/07/05/1e3938d5057f467ae.png)

- 为了实现逻辑回归，我们可以将每个特征值都乘以一个回归系数，然后把所有结果值相加，总和代入Sigmoid函数中进行分类

- Sigmoid函数的输入为z

  
  $$
  z = w_0x_0+w_1x_1+w_2x_2+...+w_nx_n
  $$

- 改为向量写法
  $$
  z = W^TX
  $$
  其中X为输入数据，W为我们要找的最佳参数为了寻找最佳参数我们使用梯度上升法

### 梯度上升

> 梯度 = 向量 = 值 + 方向
> 梯度 = 梯度值 + 梯度方向
> 爬山时，步长 + 前进方向 

- 思想：要找到函数的最大值，最好方法是沿着该函数的梯度方向进行寻找

- ![](https://www.z4a.net/images/2018/07/05/3187f7831630af342.png)

- 

- 例如：![](https://www.z4a.net/images/2018/07/05/256e4e808b78590c0.png)

- 这是一幅等高线图，我们可以把它看成一座山的等高线图，我们目标是山顶（函数最高值），初始出发点为P0，从P0开始计算梯度，然后根据此梯度上山到P1点，在P1点再计算梯度，并沿着新的梯度方向移动到P2，如此进行循环，直到到达山顶或者到达最大迭代次数，可以用如下数学公式表示

- $$
  w:=w+\alpha\nabla_Wf(w)
  $$

- 其中 $$ \nabla_Wf(w) $$ 表示$$ f(w) $$ 函数对W求导，$$ \alpha $$ 表示前进的步长

### 梯度下降

- 其实这个两个方法在此情况下本质上是相同的。关键在于代价函数（cost function）如果目标函数是损失函数，那就是最小化损失函数来求函数的最小值，就用梯度下降。 如果目标函数是似然函数（Likelihood function），就是要最大化似然函数来求函数的最大值，那就用梯度上升。在逻辑回归中， 损失函数和似然函数无非就是互为正负关系
- 只需要在迭代公式中的加法变成减法，梯度下降对应公式 $$ w:=w-\alpha\nabla_Wf(w) $$

### 工作原理

> 每个回归系数初始化为1
> 重复R次：
> 	计算整个数据集的梯度
> 	使用 步长 x 梯度 更新回归系数的向量
> 返回回归系数

## 代码

### 梯度上升

#### 更新梯度

```python
def load_data_set():
    '''
    加载数据集
    :return: 原始数据集
    '''
    data_arr = []
    label_arr = []
    f = open('testSet.txt', 'r')
    for line in f.readlines():
        # strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
        line_arr = line.strip().split()
        data_arr.append([1.0, np.float(line_arr[0]), np.float(line_arr[1])])
        label_arr.append(int(line_arr[2]))
    return data_arr, label_arr

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def grad_ascent(data_arr, class_labels):
    '''
    梯度上升
    :param data_arr:输入数组
    :param class_labels:类别标签，1*100的行向量
    :return:weight
    '''
    # 将数组变为矩阵
    data_mat = np.mat(data_arr)
    # label转化为矩阵后转置,得到label的列向量
    label_mat = np.mat(class_labels).transpose()
    # m为样本数，n为特征数
    m, n = np.shape(data_mat)
    # 步长
    alpha = 0.001
    # 最大迭代次数
    max_cycles = 500
    # weights 代表回归系数， 此处的 ones((n,1)) 创建一个长度和特征数相同的矩阵，其中的数全部都是 1
    weights = np.ones((n, 1))
    for k in range(max_cycles):
        # n*3   *  3*1  = n*1
        # 矩阵乘法
        h = sigmoid(data_mat * weights)
        error = label_mat - h
        # 0.001* (3*m)*(m*1) 表示在每一个列上的一个误差情况，最后得出 x1,x2,xn的系数的偏移量
        # alpha * dataMatrix.transpose() * error 参考：http://lib.csdn.net/article/machinelearning/35119
        weights = weights + alpha + data_mat.transpose() * error
    return weights
```

- 测试结果：![](https://www.z4a.net/images/2018/07/05/4888e2c5ec465329b.png)

#### 分析数据：画出决策边界

```python
def plot_best_fit(weights):
    '''
    可视化画出决策边界
    :param weights:
    :return:
    '''
    import matplotlib.pyplot as plt
    data_mat, label_mat = load_data_set()
    data_arr = np.array(data_mat)
    n = np.shape(data_mat)[0]
    x_cord1 = []
    y_cord1 = []
    x_cord2 = []
    y_cord2 = []
    for i in range(n):
        if int(label_mat[i]) == 1:
            x_cord1.append(data_arr[i, 1])
            y_cord1.append(data_arr[i, 2])
        else:
            x_cord2.append(data_arr[i, 1])
            y_cord2.append(data_arr[i, 2])
    fig = plt.figure()
    # 1*1的网格其中的第一个子图
    ax = fig.add_subplot(111)
    ax.scatter(x_cord1, y_cord1, s=30, color='k', marker='^')
    ax.scatter(x_cord2, y_cord2, s=30, color='red', marker='s')
    x = np.arange(-3.0, 3.0, 0.1)
    '''
    w0*x0+w1*x1+w2*x2=f(x)
    x0最开始就设置为1， x2就是我们画图的y值，而f(x)被我们磨合误差给算到w0,w1,w2身上去了
    0是两个分类（类别1和类别0）的分界处，所以令f(x) = 0
    所以： w0+w1*x+w2*y=0 => y = (-w0-w1*x)/w2   
    '''
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('x1')
    plt.ylabel('y1')
    plt.show()
```

- 结果：![](https://www.z4a.net/images/2018/07/05/51e5f541ec6d3aefc.png)
- 只分错了四个点，但是计算了300次乘法

#### 训练算法：随机梯度上升

- 梯度上升算法在每次更新回归系数时都需要遍历整个数据集，该方法在处理 100 个左右的数据集时尚可，但如果有数十亿样本和成千上万的特征，那么该方法的计算复杂度就太高了。一种改进方法是一次仅用一个样本点来更新回归系数，该方法称为 **随机梯度上升算法**

  > 所有回归系数初始化为 1 
  > 对数据集中每个样本 
  > 	计算该样本的梯度 
  > 	使用 alpha x gradient 更新回归系数值 
  > 返回回归系数值 

```python
def stoc_grad_ascent0(data_mat, class_label):
    '''
    随机梯度上升
    :param data_mat:输入数据的数据特征
    :param class_label:输入数据的类别标签
    :return: 最佳回归系数
    '''
    m, n = np.shape(data_mat)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        # sum(data_mat[i]*weights)为了求 f(x)的值， f(x)=a1*x1+b2*x2+..+nn*xn,
        # 此处求出的 h 是一个具体的数值，而不是一个矩阵
        h = sigmoid(sum(data_mat[i] * weights))
        error = class_label[i] - h
        weights = weights + alpha * error * data_mat[i]
    return weights
```

- 测试结果：![](https://www.z4a.net/images/2018/07/05/6938a617c107221f3.png)
- 分错的比较多，但是前一个是循环了500次的分类结果，所以对随机梯度下降再次修改，加入循环次数

#### 改进随机梯度上升

```python
def stoc_grad_ascent1(data_mat, class_label, num_iter = 150):
    '''
    改进版的梯度上升，使用随机的一个样本来更新回归系数
    :param data_mat: 输入数据的数据特征
    :param class_label: 输入数据的类别标签
    :param num_iter: 迭代次数
    :return: 最佳回归系数
    '''
    m, n = np.shape(data_mat)
    weights = np.ones(n)
    for j in range(num_iter):
        data_index = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            # random.uniform(x, y) 方法将随机生成下一个实数，它在[x,y]范围内,x是这个范围内的最小值，y是这个范围内的最大值。
            rand_index = int(np.random.uniform(0, len(data_index)))
            h = sigmoid(np.sum(data_mat[data_index[rand_index]] * weights))
            error = class_label[data_index[rand_index]] - h
            weights = weights + alpha * error * data_mat[data_index[rand_index]]
            del(data_index[rand_index])
    return weights
```

- 改进了三处：1.修改了alpha的值，会随着迭代次数不断减小，因为越到最高点，越需要小的步长；

  ​		      2.通过随机选取样本更新回归系数，用完之后会从列表处删除该值

  ​		     3.增加了一个迭代次数的参数，默认为150次

- 结果：![](https://www.z4a.net/images/2018/07/05/7.png)

- 可以看到效果与1相似，但所用计算量更少

### 疝气病预测病马死亡率

#### 准备数据

> 当数据有所缺失时，可以：
> 1. 使用特征均值来填补缺失
> 2. 使用特殊值来填补缺失，如-1
> 3. 忽略缺失值的样本
> 4. 使用相似样本均值填补缺失
> 5. 使用另外的机器学习算法预测缺失值
> 6. 丢弃整列缺失值

- 此处用的是处理缺失后的数据集，原数据集在http://archive.ics.uci.edu/ml/datasets/Horse+Colic

  

#### 测试算法

```python
def classify_vector(int_x, weights):
    '''
    1.将测试集的每个特征向量乘以最优化方法得到回归系数
    2.将乘积求和
    3.将结果输入到sigmoid函数中，答案大于0.5则结果为1
    :param int_x: 特征向量，features
    :param weights: 根据梯度下降得到的回归系数
    :return:
    '''
    prob = sigmoid(np.sum(int_x * weights))
    if prob > 0.5:
        return 1.0
    return 0.0

def colic_test():
    '''
    打开测试集及训练集，并对数据进行格式化处理
    :return:
    '''
    f_train = open('horseColicTraining.txt', 'r')
    f_test = open('horseColicTest.txt', 'r')
    training_set = []
    training_labels = []
    for line in f_train.readlines():
        curr_line = line.strip().split('\t')
        # len函数求的是列表内元素个数
        # 这里如果就一个空的元素，则跳过本次循环
        if len(curr_line) == 1:
            continue
        line_arr = [float(curr_line[i]) for i in range(21)]
        training_set.append(line_arr)
        training_labels.append(float(curr_line[21]))
    # 使用改进后的随机梯度下降算法 求得在此数据集上的最佳回归系数 trainWeights
    train_weights = stoc_grad_ascent1(np.array(training_set), training_labels, 500)
    error_count = 0
    num_test_vec = 0.0
    # 读取测试数据集进行测试，计算分类错误的样本条数和最终的错误率
    for line in f_test.readlines():
        num_test_vec += 1
        curr_line = line.strip().split('\t')
        if len(curr_line) == 1:
            continue
        line_arr = [float(curr_line[i]) for i in range(21)]
        if int(classify_vector(np.array(line_arr), train_weights)) != int(curr_line[21]):
            error_count += 1
    error_rate = error_count / num_test_vec
    print('The error rate is {}'.format(error_count))
    return error_rate
    
    def multi_test():
    '''
    调用colicTest() 10次并求结果的平均值
    :return:
    '''
    num_test = 10
    error_sum = 0
    for k in range(num_test):
        error_sum += colic_test()
    print('After {} iteration the average error rate is {}'.format(num_test, error_sum / num_test))

```

- 结果：![](https://www.z4a.net/images/2018/07/05/8.png)
- 因为数据有30%缺失，所以结果可以接受