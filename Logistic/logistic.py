# -*- coding:utf-8 -*-

import numpy as np

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

######################################## 从疝气病症预测病马死亡概率 #######################################################
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
    print('The error rate is {}'.format(error_rate))
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

