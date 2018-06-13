from numpy import *
import operator


# 创建数据集
def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0,0], [0,0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# 实现KNN，计算输入点与各点的欧式距离，找出距离最近的k个点，k个点中出现频率最高的类别就被判断为输入点的类别
def classify0(inX, dataSet, labels, k):
    """

    :param inX: 待分类的点
    :param dataSet: 训练数据集
    :param labels: 训练数据集分类标签
    :param k: kNN中的k值
    :return: inX的分类结果
    """
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 将datingtest2中的数据转换为numpy矩阵，类别存如list并返回
def file2matrix(filename):
    """

    :param filename: 训练集文件名称
    :return: 数据集矩阵，数据集类别标签向量
    """
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0: 3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


# 为避免有一个因素数值过大，对分类的影响超过了其他因素，故对所有因素的数值进行归一化（前提是所有的因素确实同等重要）
def autoNorm(dataSet):
    """

    :param dataSet: 待归一化的数据集
    :return: 归一化后的数据机，数据集值范围，数据集最小值
    """
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


# 找1/10的数据用classify0分类，测试准确性
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[normMat[numTestVecs:m, :]], \
                                     datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d"\
              % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f"% (errorCount/float(numTestVecs)))

