"""
@author: 梅少伟
@class: 电信研A1
@num: 20241513007
"""
import matplotlib.axis as maxis

maxis.Axis.converter = property(
    fget=lambda self: self.get_converter(),
    fset=lambda self, val: self.set_converter(val)
)

import matplotlib.pyplot as plt
import numpy as np


def loadDataSet(fileName):
    dataMat = []
    with open(fileName) as fr:
        for line in fr.readlines():
            curLine = line.strip().split('\t')
            fitLine = list(map(float, curLine))
            dataMat.append(fitLine)
    return dataMat


def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))


def randCent(dataSet, k):
    n = np.shape(dataSet)[1]
    centroids = np.asmatrix(np.zeros((k, n)))
    for j in range(n):
        minJ = np.min(dataSet[:, j])
        maxJ = np.max(dataSet[:, j])
        rangeJ = float(maxJ - minJ)
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)
    return centroids


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = np.shape(dataSet)[0]
    clusterAssment = np.asmatrix(np.zeros((m, 2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = float("inf")
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        for cent in range(k):
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = np.mean(ptsInClust, axis=0)
    return centroids, clusterAssment


def plotDataSet(filename):
    datMat = np.asmatrix(loadDataSet(filename))
    # 设置 k=2
    myCentroids, clustAssing = kMeans(datMat, 2)
    clustAssing = clustAssing.tolist()
    myCentroids = myCentroids.tolist()

    # 只需两组坐标
    xcord = [[], []]
    ycord = [[], []]
    datMat = datMat.tolist()
    m = len(clustAssing)
    for i in range(m):
        idx = int(clustAssing[i][0])
        xcord[idx].append(datMat[i][0])
        ycord[idx].append(datMat[i][1])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 绘制两类点
    ax.scatter(xcord[0], ycord[0], s=20, c='b', marker='*', alpha=0.5, label='Cluster 0')
    ax.scatter(xcord[1], ycord[1], s=20, c='r', marker='D', alpha=0.5, label='Cluster 1')
    # 绘制两个质心
    for idx, centroid in enumerate(myCentroids):
        ax.scatter(centroid[0], centroid[1], s=100, c='k', marker='+', alpha=0.8)
    plt.title('DataSet (k=2)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plotDataSet('testSet.txt')
