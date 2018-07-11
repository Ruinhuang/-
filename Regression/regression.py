# coding:utf-8

from numpy import *
import matplotlib.pyplot as plt

def load_data_set(file_name):
    '''
    加载数据
    解析以tab键分隔的文件中的浮点数
    :param file_name:输入数据
    :return:
        dataMat ：  feature 对应的数据集
        labelMat ： feature 对应的类别标签
    '''
    # 获取样本特征的总数
    num_feat = len(open(file_name).readline().split('\t')) -1
    data_mat = []
    label_mat = []
    fr = open(file_name)
    for line in fr.readlines():
        # 读取每一行
        line_arr = []
        cur_line = line.strip().split('\t')
        for i in range(num_feat):
            line_arr.append(float(cur_line[i]))
        data_mat.append(line_arr)
        label_mat.append(float(cur_line[-1]))
    return data_mat, label_mat

def stand_regres(x_arr, y_arr):
    '''
    线性回归
    :param x_arr:输入的样本数据，包含每个样本数据的 feature
    :param y_arr:对应于输入数据的类别标签，也就是每个样本对应的目标变量
    :return:
        ws：回归系数
    '''
    x_mat = mat(x_arr)
    y_mat = mat(y_arr).T
    xTx = x_mat.T * x_mat
    # 因为要用到xTx的逆矩阵，所以事先需要确定计算得到的xTx是否可逆，条件是矩阵的行列式不为0
    # linalg.det() 函数是用来求得矩阵的行列式的，如果矩阵的行列式为0，则这个矩阵是不可逆的，就无法进行接下来的运算
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (x_mat.T * y_mat)
    return ws

def stand_regres_plot():
    '''
    画图
    :return: 拟合线性图
    '''
    # x_arr: 从load_data_set得到的特征数组
    # y_arr:从load_data_set得到的特征数组
    x_arr, y_arr = load_data_set('ex0.txt')
    x_mat = mat(x_arr)
    y_mat = mat(y_arr)
    # ws: 用stand_regres计算得到的回归系数
    ws = stand_regres(x_arr, y_arr)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 绘制原始数据点
    ax.scatter(x_mat[:, 1].flatten().A[0], y_mat.T[:, 0].flatten().A[0])
    # 如果直线上的点次序混乱，绘图将会出现问题，所以对直线上的点按照升序进行排序
    x_copy = x_mat.copy()
    x_copy.sort(0)
    y_hat = x_copy * ws
    ax.plot(x_copy[:, 1], y_hat)
    plt.show()


def lwlr(test_point, x_arr, y_arr, k =0.1):
    '''
    局部加权线性回归，在待预测点附近的每个点赋予一定的权重，在子集上基于最小均方差来进行普通的回归
    :param test_point:测试点
    :param x_arr:样本的特征数据，即 feature
    :param y_arr:每个样本对应的类别标签，即目标变量
    :param k:关于赋予权重矩阵的核的一个参数，与权重的衰减速率有关；其中k是带宽参数，控制w（钟形函数）的宽窄程度，类似于高斯函数的标准差
    :return:
        testPoint * ws：数据点与具有权重的系数相乘得到的预测点

    note:
    高斯权重的公式，w = e^((x^((i))-x) / -2k^2) x为某个预测点，x^((i))为样本点
    样本点距离预测点越近，w越大，贡献的误差越大（权值越大），越远则贡献的误差越小（权值越小）
    '''
    x_mat = mat(x_arr)
    y_mat = mat(y_arr).T
    # 获得xMat矩阵的行数
    m = shape(x_mat)[0]
    weights = mat(eye((m)))
    for j in range(m):
        diff_mat = test_point - x_mat[j, :]
        weights[j, j] = exp(diff_mat * diff_mat.T / (-2.0*k**2))
    xTx = x_mat.T * (weights * x_mat)
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (x_mat.T * (weights * y_mat))
    return test_point * ws

def lwlr_test(test_arr, x_arr, y_arr, k = 0.1):
    '''
    对数据集中每个点调用 lwlr() 函数
    :param test_arr:测试样本点
    :param x_arr:样本的特征数据，即 feature
    :param y_arr:每个样本对应的类别标签
    :param k:控制核函数的衰减速率
    :return:预测点的估计值
    '''
    m = shape(test_arr)[0]
    y_hat = zeros(m)
    # 循环所有的数据点，并将lwlr运用于所有的数据点
    for i in range(m):
        y_hat[i] = lwlr(test_arr[i], x_arr, y_arr, k)
    # 返回估计值
    return y_hat

# def lwlr_for_plot(x_arr, y_arr, k = 1.0):
#     '''
#     首先将 X 排序，其余的都与lwlrTest相同，这样更容易绘图
#     :param x_arr:样本的特征数据，即 feature
#     :param y_arr:每个样本对应的类别标签,实际值
#     :param k:控制核函数的衰减速率的有关参数，这里设定的是常量值 1
#     :return:
#         yHat：样本点的估计值
#         xCopy：xArr的复制
#     '''
#     y_hat = zeros(shape(y_arr))
#     x_copy = mat(x_arr)
#     x_copy.sort(0)
#     for i in range(shape(x_arr)[0]):
#         y_hat[i] = lwlr(x_copy, x_arr, y_arr, k)
#     return y_hat, x_copy

def lwlr_plot():
    x_arr, y_arr = load_data_set('ex0.txt')
    y_hat = lwlr_test(x_arr, x_arr, y_arr, 0.003)
    x_mat = mat(x_arr)
    # argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)
    srt_ind = x_mat[:, 1].argsort(0)
    x_sort = x_mat[srt_ind][:, 0, :]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x_sort[:, 1], y_hat[srt_ind])
    ax.scatter(x_mat[:, 1].flatten().A[0], mat(y_arr).T.flatten().A[0], s=2, c='red')
    plt.show()



def rss_error(y_arr, y_hat_arr):
    '''
    分析预测误差大小
    :param y_arr:
    :param y_hat_arr:
    :return:
    '''
    return ((y_arr - y_hat_arr) ** 2).sum()

def ridge_regres(x_mat, y_mat, lam = 0.2):
    '''
    这个函数实现了给定 lambda 下的岭回归求解
    如果数据的特征比样本点还多，就不能再使用上面介绍的的线性回归和局部现行回归了，因为计算 (xTx)^(-1)会出现错误
    如果特征比样本点还多（n > m），也就是说，输入数据的矩阵x不是满秩矩阵,非满秩矩阵在求逆时会出现问题
    :param x_mat:特征数据
    :param y_mat:类别标签
    :param lam:引入的一个λ值，使得矩阵非奇异；xTx + λI(I是对角线为1的单位矩阵)
    :return:经过岭回归计算得到的回归系数
    '''
    xTx = x_mat.T * x_mat
    # 岭回归就是在矩阵 xTx 上加一个 λI 从而使得矩阵非奇异，进而能对 xTx + λI 求逆
    denom = xTx + eye(shape(x_mat)[1]) * lam
    # 检查行列式linalg.det()是否为零，即矩阵是否可逆，行列式为0的话就不可逆，不为0的话就是可逆
    if linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (x_mat.T * y_mat)
    return ws

def ridge_test(x_arr, y_arr):
    '''
    函数 ridgeTest() 用于在一组 λ 上测试结果
    :param x_arr:
    :param y_arr:
    :return:
    '''
    x_mat = mat(x_arr)
    y_mat = mat(y_arr).T
    # 对各列求均值
    y_mean = mean(y_mat, 0)
    # Y的所有的特征减去均值
    y_mat = y_mat - y_mean
    # 标准化 x，计算 xMat 平均值
    x_means = mean(x_mat, 0)
    # 然后计算 X的方差
    x_var = var(x_mat, 0)
    # 数据标准化，所有特征都减去各自的均值并除以方差
    x_mat = (x_mat - x_means) / x_var
    # 可以在 30 个不同的 lambda 下调用 ridgeRegres() 函数
    num_test_pts = 30
    # 创建30 * m 的全部数据为0 的矩阵
    w_mat = zeros((num_test_pts, shape(x_mat)[1]))
    for i in range(num_test_pts):
        ws = ridge_regres(x_mat, y_mat, exp(i - 10))
        w_mat[i, :] = ws.T
    return w_mat

def ridge_plot():
    abX, abY = load_data_set('abalone.txt')
    ridge_weights = ridge_test(abX, abY)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridge_weights)
    plt.show()