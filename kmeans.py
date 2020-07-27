import numpy as np
import matplotlib as pml
import matplotlib.pyplot as plt
import pandas as pd


# read_csv ,分隔
# iris = pd.read_csv('iris.txt', header=None)
iris = pd.read_csv('defense/20200711_5hosts_c4.csv', header=None)
# print(iris.head())
# print(iris.shape)
"""
函数功能：计算两个数据集之间的欧氏距离
输入：两个Array数据集
返回：两个数据集之间的欧式距离（此处用距离的平方和代替距离）
"""


def distEclud(arrA, arrB):
    d = arrA - arrB
    dist = np.sum(np.power(d, 2), axis=1)
    return dist


def randCent(dataSet, k):
    n = dataSet.shape[1]
    data_min = dataSet.iloc[:, :n-1].min()
    data_max = dataSet.iloc[:, :n-1].max()
    data_cent = np.random.uniform(data_min, data_max, (k, n-1))  # 返回一个min~max中随机的k行，n-1列的数据
    # print('初始质心为')
    # print(data_cent)
    return data_cent

#
# iris_cent = randCent(iris, 3)
# # print(iris_cent)



"""
函数功能：K-means聚类算法
参数说明：
    dataSet : 带标签数据集
    k : 簇的个数
    distMeas : 距离计算函数
    createCent ：随机质心生成函数
返回：
    centroids ：质心
    result_set : 所有数据划分结果
"""


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    # m数据条数，n数据维数
    m, n = dataSet.shape
    centroids = createCent(dataSet, k)
    clusterAssment = np.zeros((m, 3))  # 初始化clusterAssment，m个数组，一个数组3个元素（距离，本次迭代簇的标号，上次迭代的簇的标号）
    clusterAssment[:, 0] = np.inf  # np.inf无穷大
    clusterAssment[:, 1:3] = -1  # clusterAssment = (inf, -1, -1)第一列距离为inf，第二列、第三列初始化为-1
    # 因为clusterAssment为Array，所以使用pd.DataFrame转换成DataFrame
    result_set = pd.concat([dataSet, pd.DataFrame(clusterAssment)], axis=1, ignore_index=True)  # 将原始数据集和clusterAssment拼接起来

    # 开始更新result_set
    clusterChanged = True
    q = 0
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            # dist ：dataSet中除了标签以外的列中的数值，距离质心的距离
            dist =distMeas(dataSet.iloc[i, :n-1].values, centroids)
            result_set.iloc[i, n] = dist.min()
            result_set.iloc[i, n+1] = np.where(dist == dist.min())[0]  # np.where(dist==dist.min())[0]返回最小距离所在的dist里的索引（位置
            # 对比上次迭代的最小距离与本次迭代的最小距离是否【全部】一致
        clusterChanged = not (result_set.iloc[:, -1] == result_set.iloc[:, -2]).all()
        if clusterChanged:
            # 对簇中所有点求均值然后更新质心
            cent_df = result_set.groupby(n+1).mean()
            # 质心更新为当前质心
            centroids = cent_df.iloc[:, :n-1].values
            # result_set中的质心进行更新
            result_set.iloc[:, -1] = result_set.iloc[:, -2]
        q = q + 1
        sse = result_set.iloc[:, n].sum()
        # print('迭代次数')
        # print(q)
        # print('sse')
        # print(sse)
    return centroids, result_set, q


"""
函数功能：聚类学习曲线
参数说明：
    dataSet : 原始数据集
    cluster ：Kmeans聚类方法
    k ；簇的个数
返回：误差平方和SSE
"""


def kclearingCurve(dataSet, cluster=kMeans, k=10):
    n = dataSet.shape[1]
    SSE = []
    for i in range(1, k):
        centroids, result_set, times = cluster(dataSet, i + 1)
        SSE.append(result_set.iloc[:, n].sum())
    plt.plot(range(2, k + 1), SSE, '--o')
    plt.show()
    return SSE


def ktimesCurve(dataSet, cluster=kMeans, k=15):
    num = []
    for i in range(1, k):
        centroids, result_set, times = cluster(dataSet, i + 1)
        num.append(times)
    plt.plot(range(2, k + 1), num, '--o')
    plt.show()
    return num


iris_cent, iris_result, q = kMeans(iris, 7)

pd.set_option('display.max_rows', None)
print(iris_result)
print('sse')
print(iris_result.iloc[:, 8].sum())
# print("最终质心为：")
# print(iris_cent)
print('迭代次数')
print(q)
print('结果统计')
print(iris_result.iloc[:, -1].value_counts())



# testSet = pd.read_csv('testSet.txt', header=None, sep='\t')
# # plt.scatter(testSet.iloc[:, 0], testSet.iloc[:, 1])
# ze = pd.DataFrame(np.zeros(testSet.shape[0]).reshape(-1, 1))
# test_cent = pd.concat([testSet, ze], axis=1, ignore_index=True)
#
# test_cent, test_cluster = kMeans(testSet, 4)
# print(test_cent)
# plt.scatter(test_cluster.iloc[:, 0], test_cluster.iloc[:, 1], c=test_cluster.iloc[:, -1])
# # plt.scatter(test_cent[:, 0], test_cent[:, 1], color='red', marker='x', s=80)
# plt.show()

# test_cluster.iloc[:, 3].sum()
# kclearingCurve(iris)
# ktimesCurve(iris)
